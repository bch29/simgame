//! Implement `BackgroundObject` to allow an object to live on a background thread, where it can
//! receive actions from the main thread and send responses back.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use crossbeam::{atomic::AtomicCell, channel};

#[derive(Clone, Debug)]
pub struct ConnectionSettings {
    pub tick_receive_timeout: Duration,
    pub terminate_timeout: Duration,

    pub tick_channel_size: usize,
    pub user_channel_size: usize,
    pub response_channel_size: usize,
}

impl Default for ConnectionSettings {
    fn default() -> Self {
        Self {
            tick_receive_timeout: Duration::from_millis(5),
            terminate_timeout: Duration::from_secs(2),

            tick_channel_size: 128,
            user_channel_size: 1,
            response_channel_size: 1024,
        }
    }
}

/// A connection to an object living on a background thread. Used to send actions to the object and
/// receive its responses.
pub struct Connection<T: BackgroundObject>
where
    T::Response: Cumulative,
{
    join_handle: Option<std::thread::JoinHandle<Result<()>>>,
    response_receiver: channel::Receiver<T::Response>,
    user_sender: channel::Sender<T::UserAction>,
    tick_sender: channel::Sender<T::TickAction>,
    killed: Arc<AtomicCell<bool>>,

    settings: ConnectionSettings,
    response: T::Response,
    count_drops_total: usize,
    count_drops: usize,

    _marker: std::marker::PhantomData<T>,
}

pub trait BackgroundObject {
    type UserAction: Send + Sync + 'static;
    type TickAction: Send + Sync + 'static;
    type Response: Send + Sync + 'static;

    fn receive_user(&mut self, action: Self::UserAction) -> Result<()>;
    fn receive_tick(&mut self, action: Self::TickAction) -> Result<()>;
    fn produce_response(&mut self) -> Result<Option<Self::Response>>;
}

pub trait Cumulative {
    fn empty() -> Self;
    fn append(&mut self, other: Self) -> Result<()>;
}

struct SecondaryConnection<T: BackgroundObject> {
    user_receiver: channel::Receiver<T::UserAction>,
    tick_receiver: channel::Receiver<T::TickAction>,
    response_sender: channel::Sender<T::Response>,
    killed: Arc<AtomicCell<bool>>,
    settings: ConnectionSettings,
    object: T,
}

impl<T> Connection<T>
where
    T: BackgroundObject,
    T::Response: Cumulative,
{
    pub fn new(object: T, settings: ConnectionSettings) -> Result<Self>
    where
        T: BackgroundObject + Send + 'static,
        T::Response: Cumulative,
    {
        let killed = Arc::new(AtomicCell::new(false));
        let (user_sender, user_receiver) = channel::bounded(settings.user_channel_size);
        let (response_sender, response_receiver) =
            channel::bounded(settings.response_channel_size);
        let (tick_sender, tick_receiver) = channel::bounded(settings.tick_channel_size);

        let mut secondary = SecondaryConnection {
            user_receiver,
            response_sender,
            tick_receiver,
            killed: killed.clone(),
            settings: settings.clone(),
            object,
        };

        let join_handle = std::thread::Builder::new()
            .name("world_update_thread".into())
            .spawn(move || secondary.run())?;

        let primary = Connection {
            join_handle: Some(join_handle),
            user_sender,
            response_receiver,
            tick_sender,
            killed,

            settings,
            response: Cumulative::empty(),
            count_drops: 0,
            count_drops_total: 0,

            _marker: std::marker::PhantomData,
        };

        Ok(primary)
    }

    pub fn send_tick(&mut self, action: T::TickAction) -> Result<()> {
        match self.tick_sender.try_send(action) {
            Ok(()) => {
                if self.count_drops > 0 {
                    log::warn!("dropped {} ticks", self.count_drops);
                    self.count_drops = 0;
                }

                self.try_collect_responses()
            }
            Err(channel::TrySendError::Full(_)) => {
                if self.count_drops == 0 {
                    log::warn!("dropping ticks due to a busy background thread")
                }

                self.count_drops += 1;
                self.count_drops_total += 1;
                Ok(())
            }
            Err(channel::TrySendError::Disconnected(_)) => self.finish(),
        }
    }

    pub fn send_user(&mut self, action: T::UserAction) -> Result<()>
    where
        T::UserAction: std::fmt::Debug,
    {
        match self.user_sender.try_send(action) {
            Ok(()) => {
                if self.count_drops > 0 {
                    log::warn!("dropped {} ticks", self.count_drops);
                    self.count_drops = 0;
                }

                self.try_collect_responses()
            }
            Err(channel::TrySendError::Full(action)) => {
                log::warn!("dropped update action {:?}", action);
                self.try_collect_responses()
            }
            Err(channel::TrySendError::Disconnected(_)) => self.finish(),
        }
    }

    fn try_collect_responses(&mut self) -> Result<()> {
        loop {
            match self.response_receiver.try_recv() {
                Ok(response) => self.response.append(response)?,
                Err(channel::TryRecvError::Empty) => break Ok(()),
                Err(channel::TryRecvError::Disconnected) => break self.finish(),
            }
        }
    }

    pub fn finish(&mut self) -> Result<()> {
        self.killed.store(true);
        let mut diff_count = 0;

        let begin_time = Instant::now();

        let clean_exit = loop {
            let now = Instant::now();
            let elapsed = now.duration_since(begin_time);
            if elapsed > self.settings.terminate_timeout {
                break false;
            }

            let remaining_timeout = self.settings.terminate_timeout - elapsed;
            match self.response_receiver.recv_timeout(remaining_timeout) {
                Ok(response) => {
                    diff_count += 1;
                    self.response.append(response)?;
                }
                Err(channel::RecvTimeoutError::Disconnected) => break true,
                Err(channel::RecvTimeoutError::Timeout) => break false,
            }
        };

        log::info!(
            "Background thread killed. Clean exit: {}. Ticks dropped: {}. Final diff batch size: {}.",
            clean_exit, self.count_drops_total, diff_count
        );

        if clean_exit {
            match self
                .join_handle
                .take()
                .ok_or_else(|| anyhow!("joined background thread twice"))?
                .join()
            {
                Ok(res) => res,
                Err(err) => Err(anyhow!("background thread panicked: {:?}", err)),
            }
        } else {
            Err(anyhow!(
                "Background thread did not finish within {:?}",
                self.settings.terminate_timeout
            ))
        }
    }

    /// Returns the accumulated responses sent by the background object.
    pub fn current_response(&mut self) -> Result<&mut T::Response> {
        self.try_collect_responses()?;
        Ok(&mut self.response)
    }
}

impl<T> Drop for Connection<T>
where
    T: BackgroundObject,
    T::Response: Cumulative,
{
    fn drop(&mut self) {
        match self.finish() {
            Ok(()) => {}
            Err(end_error) => {
                for error in end_error.chain() {
                    log::error!("{}", error);
                    log::error!("========");
                }
            }
        }
    }
}

impl<T> SecondaryConnection<T>
where
    T: BackgroundObject,
{
    fn run(&mut self) -> Result<()> {
        loop {
            if self.killed.load() {
                log::info!("background thread killed");
                break;
            }

            self.send_response()?;

            if let Ok(action) = self.user_receiver.try_recv() {
                self.object.receive_user(action)?;
            }

            if let Ok(action) = self
                .tick_receiver
                .recv_timeout(self.settings.tick_receive_timeout)
            {
                self.object.receive_tick(action)?;
            }
        }

        Ok(())
    }

    fn send_response(&mut self) -> Result<()> {
        let resp = match self.object.produce_response()? {
            Some(resp) => resp,
            None => return Ok(()),
        };

        Ok(self.response_sender.send(resp)?)
    }
}
