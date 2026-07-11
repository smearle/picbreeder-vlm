NAME
     pbRender -- renders images from Picbreeder chromosomes

USAGE
      pbRender [ -id #### | -all #### | -in filePath ] options...

DESCRIPTION
     For a series (id) in the genotype repository or for another chromosome that
     is formated as a Picbreeder chromosome (in), pbRender generates an image
     by querying the CPPN for the ink value of each pixel position in the image

     The following options are available:
     -size   Specify the width and height of the the image to be rendered.
             (e.g. 128x128, 256x256, 512x512, etc.)

     -s      Creates a 'small' sized image that is 128x128 pixels.
             The options -s, -sm and -small are synonymous.

     -m      Creates a 'medium' sized image that is 256x256 pixels.
             The options -m, -med and -medium are synonymous.

     -l      Creates a 'large' sized image that is 512x512 pixels.
             The options -l, -lg and -large are synonymous.

     -xl     Creates an 'extra large' image that is 1024x1024 pixels.
             The options -xl, -xlg and -xLarge are synonymous.

     -png    Specifies the output file encoding to be PNG.

     -jpg    Specifies the output file encoding to be JPG.

     -gif    Specifies the output file encoding to be GIF

     -bmp    Specifies the output file encoding to be BMP

     -id     Specifies the series id of the published image to be rendered.
             Only the published image in the series is rendered.

     -all    Saves images for each intermediate image in the current series.
             For example, if the user that published series X made three 
             selections, then the output would be files X.1, X.2 and X.3.

     -in     Sets the path for a Picbreeder like chromosome to be rendered.

     -db     Sets the archive database to query

     -out    Sets the output directory where images file will be written.

     -show   Causes the images to be rendered on the screen instead of being
             written out as files.
     The -size, -s, -m, -l and -xl options are processed in the priority order
     shown, i.e. specifying -size nullifies the use of -s, -m, -l or -xl.

     The -png, -jpg, -gif and -bmp options are processed in the order shown,
     i.e. specifying -png nullifies the use of -jpg, -gif or -bmp.

EXAMPLES (Linux, OSX)
     The following is how to render an 128x128 pixel jpg of chromosome 94 in
     the Picbreeder repository and place it into the renderedImages directory
     
             ./pbRender.sh -id 94 -small -jpg -out renderedImages
             ./pbRender.sh -all 94 -size 256x256 -show

EXAMPLES (Windows 7)
     The following is how to render an 128x128 pixel jpg of chromosome 94 in
     the Picbreeder repository and place it into the renderedImages directory
     
             .\pbRender.bat -id 94 -small -jpg -out renderedImages
             .\pbRender.bat -all 94 -size 256x256 -show

