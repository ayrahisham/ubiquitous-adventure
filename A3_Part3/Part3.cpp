// Name: Nur Suhaira Bte Badrul Hisham
// Student ID: 5841549
// Assignment 3 Part 3

// Without this, the compiler will use the default OpenCL 2.x, 
// where some OpenCL 1.2 functions are deprecated.
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

// By default, OpenCL disables exceptions.This line enables it.
#define __CL_ENABLE_EXCEPTIONS

// The headers from the C++ standard library and STL that are used
#include <iostream>
#include <fstream> // For file reading
#include <string>

// Including the OpenCL header. 
// Depending on which Operating System (OS) 
// we are using, this checks whether we are running the program on Mac OS 
// or other OS :
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// Include a user defined header file :
#include "common.h"
#include "bmpfuncs.h"

// Having to type std:: and cl:: prefixes in front of the C++
// standard library or OpenCL functions
// using namespace std;
// using namespace cl;

// To get more reliable results, run the kernel at least 1000 times for each case 
// and calculate the average time
#define NUM_ITERATIONS 1000 

int main()
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// To perform profiling, an event must be associated with the command to profile
	cl::Event profileEvent;
	cl_ulong timeStart, timeEnd, timeTotal;

	// declare data and memory objects
	unsigned char* inputImage;
	unsigned char* outputImage;
	int imgWidth, imgHeight, imageSize;

	cl::ImageFormat imgFormat;
	cl::Image2D inputImgBuffer; // original image input
	cl::Image2D inputImgBuffer2; // another image input
	cl::Image2D outputImgBuffer;
	cl::Buffer wsizeBuffer;			// window size buffer

	cl::Buffer sizeBuffer;			// image size buffer
	cl::Buffer dataBuffer;			// to store luminance values
	cl::Buffer resultsRWBuffer;		// to read results (reading) from horizontal then write new results from vertical

	cl_int size;
	string type = "";

	const char *outputfile1b = "image1b.bmp"; // an image where the glowing pixels are kept, while the rest are set to black.
	const char *outputfile1cH = "image1cH.bmp"; // an image where the images undergoes a horizontal blur pass
	const char *outputfile1cV = "image1cV.bmp"; // an image where the images undergoes a vertical blur pass
	const char *outputfile1d = "image1d.bmp"; // an image where the images with bloom effect

	cl_int imagesize[2] = { 0 };
	cl_int pixelcount = 0; // number of pixels in image

	try
	{
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program("Device not selected.");
		}

		// create a context from device
		context = cl::Context(device);

		// build the program
		if (!build_program(&program, &context, "Part3.cl"))
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		cl_device_type usertype = device.getInfo<CL_DEVICE_TYPE>();
		if (usertype == CL_DEVICE_TYPE_CPU)
		{
			type = "CPU";
		}
		else if (usertype == CL_DEVICE_TYPE_GPU)
		{
			type = "GPU";
		}

		// create command queue
		// Note that when the command queue was created, the profiling flag was enabled
		queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
		std::cout << "Creating a command queue..." << std::endl;

		int choice;
		while (true)
		{
			std::cout << "\nChoose an image in folder: " << std::endl;
			std::cout << "\t1. bunnycity1.bmp" << std::endl;
			std::cout << "\t2. bunnycity2.bmp" << std::endl;
			std::cout << "Choice: ";
			std::cin >> choice;
			if (choice == 1 || choice == 2)
			{
				break;
			}
		}
		std::cin.clear();
		std::cin.ignore(10, '\n');

		// read input image
		if (choice == 1)
		{
			std::cout << "\nReading input image 'bunnycity1' to RGBA..." << std::endl;
			inputImage = read_BMP_RGB_to_RGBA("bunnycity1.bmp", &imgWidth, &imgHeight);
		}
		else
		{
			std::cout << "\nReading input image 'bunnycity2' to RGBA..." << std::endl;
			inputImage = read_BMP_RGB_to_RGBA("bunnycity2.bmp", &imgWidth, &imgHeight);
		}

		float luminance;
		while (true)
		{
			std::cout << "\nEnter threshold value for luminance computation (enter 0 for default value): ";
			std::cin >> luminance;
			if (luminance == 0) 
			{
				if (choice == 1) // bunnycity1.bmp image
				{
					luminance = 101.45 / 255;
					break;
				}
				else // choice == 2 (bunnycity2.bmp)
				{
					luminance = 116.56 / 255;
					break;
				}
			}
			else if (luminance > 0 && luminance < 256) // luminance from 1 to 155
			{
				luminance /= 255.0; // convert to float
				break;
			}
			else // luminance negative values and > 255
			{
				std::cout << "\tPlease enter a valid threshold value" << std::endl;
			}
		}
		std::cin.clear();
		std::cin.ignore(10, '\n');
		std::cout << "You have chosen to use " << luminance << " as the default threshold value.\n" << std::endl;

		// allocate memory for output image
		imageSize = imgWidth * imgHeight * 4;
		outputImage = new unsigned char[imageSize];

		// image format
		imgFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);

		// create image objects
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		dataBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float), &luminance);
		sizeBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * 2);

		std::cout << "===============Convert a colour image into a greyscale image===============\n" << std::endl;

		// create a kernel
		std::cout << "Creating a kernel to keep the glowing pixels" << std::endl;
		kernel = cl::Kernel(program, "gray_scale");

		// set kernel arguments
		std::cout << "\tSetting kernel arguments to keep glowing pixels..." << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, dataBuffer);
		kernel.setArg(3, sizeBuffer);

		// enqueue kernel
		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imgWidth, imgHeight);

		timeTotal = 0;

		for (int i = 0; i < NUM_ITERATIONS; i++)
		{
			// When the kernel enqueuing command is called, the profileEvent is associated with it (the last argument):
			queue.enqueueNDRangeKernel(kernel, offset, globalSize, cl::NullRange, NULL, &profileEvent);
			queue.finish(); // blocks the program from continuing until all commands in the command queue have completed

			// The following obtains the start and end time stamps of when the command started 
			// and ended its execution
			// Subtracting the start from the end time will give how long the command took to execute
			timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

			timeTotal += timeEnd - timeStart;
		}

		std::cout << "\tKernel enqueued for execution..." << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;

		// enqueue command to read image from device to host memory
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imgWidth;
		region[1] = imgHeight;
		region[2] = 1;

		// getting the imagesize
		queue.enqueueReadBuffer(sizeBuffer, CL_TRUE, 0, sizeof(cl_int) * 2, &imagesize[0]);
		pixelcount = (imagesize[0] * imagesize[1]);

		std::cout << "\tReading the results of a gray image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		std::cout << "\tWriting output image '" << outputfile1b << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB (outputfile1b, outputImage, imgWidth, imgHeight);

		std::cout << "'" << outputfile1b << "' is successfully created\n" << std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		std::cout << "=====================Undergoes a horizontal blur pass======================\n" << std::endl;

		// display user menu
		int windowsize = 1;
		std::cout << "Available window sizes:" << std::endl;
		std::cout << "\t1. 3x3" << std::endl;
		std::cout << "\t2. 5x5" << std::endl;
		std::cout << "\t3. 7x7" << std::endl;
		std::cout << "Enter the size of the Gaussian blur filter: ";
		std::cin >> windowsize;

		switch (windowsize)
		{
		case 1: size = 3;
			std::cout << "You have chosen to filter using a window size of 3x3" << std::endl;
			break;
		case 2: size = 5;
			std::cout << "You have chosen to filter using a window size of 5x5" << std::endl;
			break;
		default: size = 7;
			std::cout << "You have chosen to filter using a window size of 7x7" << std::endl;
		}

		// create a kernel
		std::cout << "\nCreating a kernel for horizontal pass" << std::endl;
		kernel = cl::Kernel(program, "horizontal_pass");

		// update input buffer with the output image from previous kernel
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		
		// update the size buffer to become read only for the window size after retrieving imagesize from previous kernel
		sizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &size);

		// set kernel arguments
		std::cout << "Setting kernel arguments for horizontal pass with window size of " << size << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, sizeBuffer);

		for (int i = 0; i < NUM_ITERATIONS; i++)
		{
			// When the kernel enqueuing command is called, the profileEvent is associated with it (the last argument):
			queue.enqueueNDRangeKernel(kernel, offset, globalSize, cl::NullRange, NULL, &profileEvent);
			queue.finish(); // blocks the program from continuing until all commands in the command queue have completed

			// The following obtains the start and end time stamps of when the command started 
			// and ended its execution
			// Subtracting the start from the end time will give how long the command took to execute
			timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

			timeTotal += timeEnd - timeStart;
		}

		std::cout << "\tKernel enqueued for execution..." << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;

		// enqueue command to read image from device to host memory
		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		std::cout << "\tWriting output image '" << outputfile1cH << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(outputfile1cH, outputImage, imgWidth, imgHeight);

		std::cout << "'" << outputfile1cH << "' is successfully created\n" << std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		std::cout << "======================Undergoes a vertical blur pass=======================\n" << std::endl;

		// create a kernel
		std::cout << "Creating a kernel for vertical pass" << std::endl;
		kernel = cl::Kernel(program, "vertical_pass");

		// update input buffer with the output image from previous kernel
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		// set kernel arguments
		std::cout << "Setting kernel arguments for vertical pass with window size of " << size << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, sizeBuffer);

		for (int i = 0; i < NUM_ITERATIONS; i++)
		{
			// When the kernel enqueuing command is called, the profileEvent is associated with it (the last argument):
			queue.enqueueNDRangeKernel(kernel, offset, globalSize, cl::NullRange, NULL, &profileEvent);
			queue.finish(); // blocks the program from continuing until all commands in the command queue have completed

			// The following obtains the start and end time stamps of when the command started 
			// and ended its execution
			// Subtracting the start from the end time will give how long the command took to execute
			timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

			timeTotal += timeEnd - timeStart;
		}

		std::cout << "\tKernel enqueued for execution..." << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;

		// enqueue command to read image from device to host memory
		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		std::cout << "\tWriting output image '" << outputfile1cV << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(outputfile1cV, outputImage, imgWidth, imgHeight);

		std::cout << "'" << outputfile1cV << "' is successfully created\n" << std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		std::cout << "====================Form an image with a bloom effect======================\n" << std::endl;

		// create a kernel
		std::cout << "Creating a kernel to add bloom effect" << std::endl;
		kernel = cl::Kernel(program, "bloom_effect");

		// update input buffer with the original image
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);

		// update input buffer with the output image from previous kernel
		inputImgBuffer2 = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		// set kernel arguments
		std::cout << "Setting kernel arguments for image bloom effect" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, inputImgBuffer2);
		kernel.setArg(2, outputImgBuffer);

		for (int i = 0; i < NUM_ITERATIONS; i++)
		{
			// When the kernel enqueuing command is called, the profileEvent is associated with it (the last argument):
			queue.enqueueNDRangeKernel(kernel, offset, globalSize, cl::NullRange, NULL, &profileEvent);
			queue.finish(); // blocks the program from continuing until all commands in the command queue have completed

			// The following obtains the start and end time stamps of when the command started 
			// and ended its execution
			// Subtracting the start from the end time will give how long the command took to execute
			timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

			timeTotal += timeEnd - timeStart;
		}

		std::cout << "\tKernel enqueued for execution..." << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;

		// enqueue command to read image from device to host memory
		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		std::cout << "\tWriting output image '" << outputfile1d << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(outputfile1d, outputImage, imgWidth, imgHeight);

		std::cout << "'" << outputfile1d << "' is successfully created\n" << std::endl;

		// deallocate memory
		free(inputImage);
		free(outputImage);
	}
	catch (cl::Error e)
	{
		handle_error(e);
	}

	system("pause");

	return 0;
}