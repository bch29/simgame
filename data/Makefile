# This Makefile converts gimp source files to PNGs

gimp = gimp

block_tex_names = rock.png leaf.png wood.png dirt.png
block_textures = $(patsubst %,resources/tex/block/%,$(block_tex_names))

.PHONY: default

default: $(block_textures)

clean:
	rm -vf $(block_textures)

resources/%.png: resources_src/%.xcf
	echo '(gimp-to-png "$<" "$@")(gimp-quit 0)' | cat gimp_to_pdf.scm - | $(gimp) -i -b -
