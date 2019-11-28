/* Write a parallel program to convert the RGB values in an image to luminance values 
(this approach is used to convert a colour image into a greyscale image). 
For each pixel, calculate:
Luminance = 0.299*R + 0.587*G + 0.114*B
Save the luminance image into a 24-bit BMP file. To do this, set the RGB values of each
pixel to the luminance value.*/
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__kernel void gray_scale(read_only image2d_t src_image,
					write_only image2d_t dst_image,
					__global float *results,
					__global int2 *size) 
{

   /* Get work-item’s row and column position */
   int2 coord  = (int2)(get_global_id(0), get_global_id(1));

   /* Get the input image size */
   int2 imagesize = get_image_dim (src_image); 

   /* Accumulated pixel value */
   float4 luminance = 0.0;

   float4 pixel;
   float4 black = 0.0;
   float4 white = 1.0;

	/* Read value pixel from the image */
	pixel = read_imagef (src_image, sampler, coord);

	/* For each pixel, calculate: luminance = 0.299*R + 0.587*G + 0.114*B */
	pixel.x *= 0.299;
	pixel.y *= 0.587;
	pixel.z *= 0.114;
	
	/* Cap the pixel value to 0 if < 0 and 1 if > 1,
	the bit will flip to the other side (white to black) & (black to white) */
	if (all (pixel > white)) // if color is greater than 1.0
	{
		pixel = white; // keep it as white
	}
	else if (all (pixel < black)) // if color is lesser than black
	{
		pixel = black; // keep it black
	}

	luminance += pixel.x + pixel.y + pixel.z;
	
	/* Write new pixel value to output */
	write_imagef (dst_image, coord, luminance);

	pixel.xyz = luminance.xyz;
	int width = get_image_width (src_image);
	int index  = width * coord.x + coord.y;
	results[index] = (pixel.x + pixel.y + pixel.z) / 3;

	*size = imagesize;
}

__kernel void reduction_vector (__global float4* data,
								 __local float4* partial_avg)
{
   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_avg[lid] = data[get_global_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) 
   {
      if (lid < i) 
	  {
         partial_avg[lid] += partial_avg[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) 
   {
		data[get_group_id(0)] = partial_avg[0];
   }
}

__kernel void reduction_complete (__global float4* data,
								  __local float4* partial_avg, 
							 	  __global float* results)
{

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_avg[lid] = data[get_local_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for (int i = group_size/2; i>0; i >>= 1) 
   {
      if(lid < i) 
	  {
         partial_avg[lid] += partial_avg[lid + i];
      }

      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) 
   {
      *results = partial_avg[0].s0 + partial_avg[0].s1 +
             partial_avg[0].s2 + partial_avg[0].s3;
   }
}