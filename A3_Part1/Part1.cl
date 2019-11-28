/* Accepts a colour image and outputs the results of a filtered image using Gaussian blurring 
(use a single kernel function) should be able to
handle different window sizes based on the weights provided above */
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

// 3x3 Blurring filter
__constant float BlurringN3Filter[9] = {0.077847, 0.123317, 0.077847,
										0.123317, 0.195346, 0.123317,
										0.077847, 0.123317, 0.077847};

// 5x5 Blurring filter
__constant float BlurringN5Filter[25] = {0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
										 0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
										 0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
										 0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
									     0.003765, 0.015019, 0.023792, 0.015019, 0.003765};

// 7x7 Blurring filter
__constant float BlurringN7Filter[49] = {0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036,
										 0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
										 0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
										 0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291,
										 0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
										 0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
										 0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036};

// 3x3 Blurring filter
__constant float Blurring3x3Filter[3] = {0.27901, 0.44198, 0.27901};

// 5x5 Blurring filter
__constant float Blurring5x5Filter[5] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};

// 7x7 Blurring filter
__constant float Blurring7x7Filter[7] = {0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598};

__kernel void simple_conv (read_only image2d_t src_image,
					       write_only image2d_t dst_image,
					       __global float4 *results,
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
   float4 pixel = (float4) (0.0);

   switch (wsize)
   {
		case 3: /* Iterate over the rows */
			   for(int i = -1; i <= 1; i++) 
			   {
				  coord.y =  row + i;

				  /* Iterate over the columns */
				  for(int j = -1; j <= 1; j++) 
				  {
					 coord.x = column + j;

					 /* Read value pixel from the image */ 		
					 pixel = read_imagef(src_image, sampler, coord);

					 /* Acculumate weighted sum */ 		
					 sum.xyz += pixel.xyz * BlurringN3Filter[filter_index++];
				  }
			   }
			   break;
		case 5: /* Iterate over the rows */
			   for(int i = -1; i <= 3; i++) 
			   {
				  coord.y =  row + i;

				  /* Iterate over the columns */
				  for(int j = -1; j <= 3; j++) 
				  {
					 coord.x = column + j;

					 /* Read value pixel from the image */ 		
					 pixel = read_imagef(src_image, sampler, coord);

					 /* Acculumate weighted sum */ 		
					 sum.xyz += pixel.xyz * BlurringN5Filter[filter_index++];
				  }
				}
				break;
		case 7: /* Iterate over the rows */
			   for(int i = -1; i <= 5; i++) 
			   {
				  coord.y =  row + i;

				  /* Iterate over the columns */
				  for(int j = -1; j <= 5; j++) 
				  {
					 coord.x = column + j;

					 /* Read value pixel from the image */ 		
					 pixel = read_imagef(src_image, sampler, coord);

					 /* Acculumate weighted sum */ 		
					 sum.xyz += pixel.xyz * BlurringN7Filter[filter_index++];
				  }
			   } 
	}

	/* Write new pixel value to output */
	coord = (int2)(column, row); 
	write_imagef(dst_image, coord, sum);

	*results = sum;
}

__kernel void horizontal_pass (read_only image2d_t src_image,
							write_only image2d_t dst_image,
							__global float4 *results,
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

   *results = sum;
}

__kernel void vertical_pass (read_only image2d_t src_image,
							write_only image2d_t dst_image,
							__global float4 *results,
							__global int *size) 
{
	int wsize = *size;

   // Retrieve from horizontal pass results
   float4 sumH = *results;

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

   *results = sumV;
}