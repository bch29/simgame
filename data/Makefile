# This Makefile converts gimp source files to PNGs

gimp = gimp

voxel_tex_names = rock.png leaf.png wood.png dirt.png grass_top.png grass_side.png
entity_tex_names = front.png back.png left.png right.png top.png bottom.png
voxel_textures = $(patsubst %,resources/tex/voxel/%,$(voxel_tex_names))
entity_textures = $(patsubst %,resources/tex/entity/%,$(entity_tex_names))

.PHONY: default

default: $(voxel_textures) $(entity_textures)

clean:
	rm -vf $(voxel_textures) $(entity_textures)

resources/%.png: resources_src/%.xcf
	echo '(gimp-to-png "$<" "$@")(gimp-quit 0)' | cat gimp_to_pdf.scm - | $(gimp) -i -b -
