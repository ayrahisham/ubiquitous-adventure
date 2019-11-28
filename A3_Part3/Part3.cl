__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

// 3x3 Blurring filter
__constant float Blurring3x3Filter[3] = {0.27901, 0.44198, 0.27901};

// 5x5 Blurring filter
__constant float Blurring5x5Filter[5] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};

// 7x7 Blurring filter
__constant float Blurring7x7Filter[7] = {0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598};

/* The image in Fig. 1(b) shows an image where the glowing pixels are kept, 
while the rest are set to black. 
For this assignment, use the average luminance value that you obtained from
your program in Part 2 as the default threshold value, 
i.e. pixels above the average luminance value are kept, 
while pixels below the average luminance value are set to black
*/
__kernel void gray_scale(read_only image2d_t src_image,
							write_only image2d_t dst_image,
							__global float *data,
							__global int2 *size) 
{

   /* Get work-item’s row and column position */
   int2 coord  = (int2)(get_global_id(0), get_global_id(1));

   /* Get the input image size */
   int2 imagesize = get_image_dim (src_image); 

   /* Using threshold value */
   float threshold = *data;
	
	 /* Accumulated pixel value */
   float4 luminance = 0.0;

   float4 pixel;
   float4 black = 0.0;

	/* Read value pixel from the image */
	pixel = read_imagef (src_image, sampler, coord);
	
	/* Cap the pixel value to 0 if < 0 and 1 if > 1,
	the bit will flip to the other side (white to black) & (black to white) */
	if ((pixel.x + pixel.y + pixel.z) / 3 < threshold) // pixels below the average luminance value are set to black
	{
		pixel = black; // keep it black
	}

	/* Write new pixel value to output */
	write_imagef (dst_image, coord, pixel);

	*size = imagesize;
}

/*The image undergoes a horizontal blur pass, then a vertical blur pass*/

__kernel void horizontal_pass (read_only image2d_t src_image,
							  write_only image2d_t dst_image,
							  __global int *size) 
{
	int wsize = *size;

	 /* Get work-item’s row and column position */
   int column = get_global_id(0); 
   int row = get_global_id(1);

   /* Accumulated pixel value */
   float4 sum = (float4)(0.0);

   /* Filter's current index */
   int filter_index =  0;

   int2 coord;
   float4 pixel;

   switch (wsize)
   {
		case 3: /* Iterate over row */
				coord.y =  row ;

				/* Iterate over the columns */
				for(int j = -1; j <= 1; j++) 
				{
					coord.x = column + j;
		 
					/* Read value pixel from the image */ 		
					pixel = read_imagef(src_image, sampler, coord);
						 
					/* Acculumate weighted sum */ 
					sum.xyz += pixel.xyz * Blurring3x3Filter[filter_index++];
				}
			   break;
		case 5: /* Iterate over row */
			   coord.y =  row;

			   /* Iterate over the columns */
			  for(int j = -1; j <= 3; j++) 
			  {
					coord.x = column + j;
		 
					/* Read value pixel from the image */ 	
					pixel = read_imagef(src_image, sampler, coord);
			   
					/* Acculumate weighted sum */ 
				   sum.xyz += pixel.xyz * Blurring5x5Filter[filter_index++];

			   }
			   break;
		case 7: /* Iterate over row */
			   coord.y =  row;

				/* Iterate over the columns */
				for(int j = -1; j <= 5; j++) 
				{
					 coord.x = column + j;
		 
					/* Read value pixel from the image */ 		
					pixel = read_imagef(src_image, sampler, coord);
				
				   /* Acculumate weighted sum */ 		
				   sum.xyz += pixel.xyz * Blurring7x7Filter[filter_index++];
				}
	}

   /* Write new pixel value to output */
   coord = (int2)(column, row); 
   write_imagef (dst_image, coord, sum);
}

__kernel void vertical_pass (read_only image2d_t src_image,
							write_only image2d_t dst_image,
							__global int *size) 
{
	int wsize = *size;

	 /* Get work-item’s row and column position */
   int column = get_global_id(0); 
   int row = get_global_id(1);

    /* Accumulated pixel value */
   float4 sumV = (float4)(0.0);

   /* Filter's current index */
   int filter_index =  0;

   int2 coord;
   float4 pixel = (float4) (0.0);

   switch (wsize)
   {
		   case 3: /* Iterate over column */
					coord.x = column;

				   /* Iterate over the rows */
				   for(int j = -1; j <= 1; j++) 
				   {
						coord.y =  row + j;

						/* Read value pixel from the image */ 		
						pixel = read_imagef(src_image, sampler, coord);

					  /* Acculumate weighted sum */ 			
					  sumV.xyz += pixel.xyz * Blurring3x3Filter[filter_index++];	  
				   }
				   break;
			case 5:  /* Iterate over column */
				   coord.x = column;
					
					/* Iterate over the rows */
					for(int j = -1; j <= 3; j++) 
					{
						coord.y =  row + j;

						/* Read value pixel from the image */ 		
						pixel = read_imagef(src_image, sampler, coord);

					  /* Acculumate weighted sum */  		
					  sumV.xyz += pixel.xyz * Blurring5x5Filter[filter_index++];
					}
					break;
			case 7:  /* Iterate over column */
					coord.x = column;
					
					/* Iterate over the rows */
					for(int j = -1; j <= 5; j++) 
					{
						coord.y =  row + j;

						/* Read value pixel from the image */ 		
						pixel = read_imagef(src_image, sampler, coord);

					  /* Acculumate weighted sum */  		
					  sumV.xyz += pixel.xyz * Blurring7x7Filter[filter_index++];
					}
	}

   /* Write new pixel value to output */
   coord = (int2)(column, row); 
   write_imagef(dst_image, coord, sumV);
}

__kernel void bloom_effect (read_only image2d_t ori_image,
							read_only image2d_t src_image,
							write_only image2d_t dst_image) 
{

   /* Get work-item’s row and column position */
   int2 coord  = (int2)(get_global_id(0), get_global_id(1)); 
	
	 /* Accumulated pixel value */
   float4 sum = 0.0;

   float4 pixelORI;
   float4 pixelSRC;
   float white = 1.0; // since addition of pixels might exceed more than 1 but not less than 0

   /* Read value pixel from the ori image */
	pixelORI = read_imagef (ori_image, sampler, coord);

	/* Read value pixel from the src image */
	pixelSRC = read_imagef (src_image, sampler, coord);

	sum.x = pixelORI.x + pixelSRC.x;
	sum.y = pixelORI.y + pixelSRC.y;
	sum.z = pixelORI.z + pixelSRC.z;

	/* Cap the pixel value to 0 if < 0 and 1 if > 1,
	the bit will flip to the other side (white to black) & (black to white) */
	if ((sum.x + sum.y + sum.z) / 3 > white) // pixels above are set to white
	{
		sum.xyz = white; // keep it white
	}

	/* Write new pixel value to output */
	write_imagef (dst_image, coord, sum);
}