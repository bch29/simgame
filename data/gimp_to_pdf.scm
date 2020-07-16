(define (gimp-to-png in-filename out-filename)
  (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE in-filename in-filename)))
         (drawable (car (gimp-image-get-active-layer image))))
    (file-png-save 
      RUN-NONINTERACTIVE
      image
      drawable
      out-filename out-filename
      0
      7 ;; compression
      0 ;; bkgd
      0 ;; gama
      0 ;; ofs
      0 ;; phys
      0 ;; time
      )))
