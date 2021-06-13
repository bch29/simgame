# This Makefile converts gimp source files to PNGs

gimp = gimp

voxel_tex_names = rock.png leaf.png wood.png dirt.png grass_top.png grass_side.png
voxel_textures = $(patsubst %,tex/voxel/%,$(voxel_tex_names))

test_tex_names = front.png back.png left.png right.png top.png bottom.png
head_tex_names = head_front.png head_back.png head_left.png head_right.png head_top.png head_bottom.png
test_textures = $(patsubst %,tex/entity/%,$(entity_tex_names))
dwarf_textures = $(patsubst %,tex/entity/dwarf/%,$(head_tex_names))
goblin_textures = $(patsubst %,tex/entity/goblin/%,$(head_tex_names))
entity_textures = $(test_textures) $(dwarf_textures) $(goblin_textures)

all_textures = $(voxel_textures) $(entity_textures)

.PHONY: default

default: $(all_textures)

clean:
	rm -vf $(all_textures)

%.png: src/%.xcf
	echo '(gimp-to-png "$<" "$@")(gimp-quit 0)' | cat gimp_to_pdf.scm - | $(gimp) -i -b -
