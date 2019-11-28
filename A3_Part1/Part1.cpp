// Name: Nur Suhaira Bte Badrul Hisham
// Student ID: 5841549
// Assignment 3 Part 1

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
	int imgWidth = 0;
	int imgHeight = 0;
	int imageSize;

	cl::ImageFormat imgFormat;
	cl::Image2D inputImgBuffer;
	cl::Image2D outputImgBuffer;
	cl::Buffer wsizeBuffer;			// window size buffer

	cl::Buffer resultsWBuffer;		// to store results (writing)
	cl::Buffer resultsRWBuffer;		// to read results (reading) from horizontal then write new results from vertical

	cl_float resultsN[4] = {0.0}; // allocate memory for results from Naive approach
	cl_float results2P[4] = {0.0}; // // allocate memory for results from 2-pass approach
	string type = "";
	cl_int size = 3;		// store window size (default = 3)
	cl_float tempResultsH[4] = { 0.0 };
	cl_float tempResultsV[4] = { 0.0 };

	const char *useroutputNfile = "useroutputN.bmp"; // user output file based on user's window size using naive
	const char *useroutput2Pfile = "useroutput2P.bmp"; // user output file based on user's window size using 2-pass
	const char *outputN3file = "outputN3x3.bmp"; // output image using 3x3 Naive approach
	const char *output2PH3x3file = "output2PH3x3.bmp"; // output image using 3x3 2 pass (horizontal) approach
	const char *output2P3x3file = "output2P3x3.bmp"; // output image using 3x3 2 pass approach
	const char *outputN5file = "outputN5x5.bmp"; // output image using 5x5 Naive approach
	const char *output2PH5x5file = "output2PH5x5.bmp"; // output image using 5x5 2 pass (horizontal) approach
	const char *output2P5x5file = "output2P5x5.bmp"; // output image using 5x5 2 pass approach
	const char *outputN7file = "outputN7x7.bmp"; // output image using 7x7 Naive approach
	const char *output2PH7x7file = "output2PH7x7.bmp"; // output image using 7x7 2 pass (horizontal) approach
	const char *output2P7x7file = "output2P7x7.bmp"; // output image using 7x7 2 pass approach

	try
	{
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program ("Device not selected.");
		}

		// create a context from device
		context = cl::Context(device);

		// build the program
		if (!build_program (&program, &context, "Part1.cl"))
		{
			// if OpenCL program build error
			quit_program ("OpenCL program build error.");
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
		queue = cl::CommandQueue (context, device, CL_QUEUE_PROFILING_ENABLE);
		std::cout << "Creating a command queue..." << std::endl;

		int choice;
		while (true)
		{
			std::cout << "Choose an image in folder to convert grayscale: " << std::endl;
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
			std::cout << "Reading input image 'bunnycity1' to RGBA...\n" << std::endl;
			inputImage = read_BMP_RGB_to_RGBA("bunnycity1.bmp", &imgWidth, &imgHeight);
		}
		else
		{
			std::cout << "Reading input image 'bunnycity2' to RGBA...\n" << std::endl;
			inputImage = read_BMP_RGB_to_RGBA("bunnycity2.bmp", &imgWidth, &imgHeight);
		}

		// allocate memory for output image
		imageSize = imgWidth * imgHeight * 4;
		outputImage = new unsigned char[imageSize];

		// image format
		imgFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);

		// create image objects
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)inputImage);
		outputImgBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		resultsWBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * 4);

		// create a kernel
		std::cout << "Creating a kernel to run filter using Naive approach" << std::endl;
		kernel = cl::Kernel(program, "simple_conv");

		// display user menu
		int userinput = 1;
		std::cout << "\t1. 3x3" << std::endl;
		std::cout << "\t2. 5x5" << std::endl;
		std::cout << "\t3. 7x7" << std::endl;
		std::cout << "Enter window size: ";
		std::cin >> userinput;

		switch (userinput)
		{
			case 1: size = 3;
				std::cout << "\nYou have chosen to filter using a window size of 3x3" << std::endl;
				break;
			case 2: size = 5;
				std::cout << "\nYou have chosen to filter using a window size of 5x5" << std::endl;
				break;
			case 3: size = 7;
				std::cout << "\nYou have chosen to filter using a window size of 7x7" << std::endl;
		}

		// enqueue kernel
		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imgWidth, imgHeight);

		// enqueue command to read image from device to host memory
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imgWidth;
		region[1] = imgHeight;
		region[2] = 1;

		// set kernel arguments
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsWBuffer);
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
		queue.enqueueReadBuffer(resultsWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &resultsN[0]);
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);
		write_BMP_RGBA_to_RGB(useroutputNfile, outputImage, imgWidth, imgHeight);
		std::cout << "'" << useroutputNfile << "' is successfully created in window size of " << size << '\n' << std::endl;

		// create a kernel
		std::cout << "Creating a kernel to run filter using 2-pass approach" << std::endl;
		kernel = cl::Kernel(program, "horizontal_pass");

		// display user menu
		userinput = 1;
		std::cout << "\t1. 3x3" << std::endl;
		std::cout << "\t2. 5x5" << std::endl;
		std::cout << "\t3. 7x7" << std::endl;
		std::cout << "Enter window size: ";
		std::cin >> userinput;

		switch (userinput)
		{
		case 1: size = 3;
			std::cout << "\nYou have chosen to filter using a window size of 3x3" << std::endl;
			break;
		case 2: size = 5;
			std::cout << "\nYou have chosen to filter using a window size of 5x5" << std::endl;
			break;
		case 3: size = 7;
			std::cout << "\nYou have chosen to filter using a window size of 7x7" << std::endl;
		}

		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsWBuffer);
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);
		
		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		kernel = cl::Kernel(program, "vertical_pass");

		// The result will then undergo a second pass in the vertical direction
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);

		// set kernel arguments
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsRWBuffer);
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

		// enqueue command to read image from device to host memory
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		write_BMP_RGBA_to_RGB(useroutput2Pfile, outputImage, imgWidth, imgHeight);

		std::cout << "'" << useroutput2Pfile << "' is successfully created in window size of " << size << '\n' << std::endl;

		// ========================================= 3x3 Naive approach ===================================================
		std::cout << "===============3x3 Naive Approach===============\n" << std::endl;

		// set kernel arguments
		std::cout << "Setting kernel arguments for 3x3 image convolution" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsWBuffer);
		size = 3; // window size 3x3
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		std::cout << "\tReading the result from the 3x3 Naive approach..." << std::endl;
		queue.enqueueReadBuffer(resultsWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &resultsN[0]);

		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		std::cout << "\tWriting output image '" << outputN3file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB (outputN3file, outputImage, imgWidth, imgHeight);

		std::cout << "'" << outputN3file << "' is successfully created" << std::endl;
		
		// Compare the kernel execution times based on the different window sizes, 
		// and whether it was run on a multicore CPU or GPU
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Profiling Kernel Execution Time" << std::endl;
		std::cout << "\tCL_DEVICE_NAME        : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tCL_DEVICE_TYPE        : " << type << std::endl;
		std::cout << "\tApproach              : Naive" << std::endl;
		std::cout << "\tWindow size           : 3x3" << std::endl;
		std::cout << "\tResult		      : " << *resultsN << std::endl;
		std::cout << "\tAverage Execution Time: " << timeTotal / NUM_ITERATIONS << '\n' << std::endl;
		
		// ========================================= 5x5 Naive approach ===================================================
		std::cout << "===============5x5 Naive Approach===============\n" << std::endl;

		// set kernel arguments
		std::cout << "Setting kernel arguments for 5x5 image convolution" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsWBuffer);
		size = 5; // window size 5x5
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		std::cout << "\tReading the result from the 5x5 Naive approach..." << std::endl;
		queue.enqueueReadBuffer(resultsWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &resultsN[0]);

		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		std::cout << "\tWriting output image '" << outputN5file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(outputN5file, outputImage, imgWidth, imgHeight);

		std::cout << "'" << outputN5file << "' is successfully created" << std::endl;

		// Compare the kernel execution times based on the different window sizes, 
		// and whether it was run on a multicore CPU or GPU
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Profiling Kernel Execution Time" << std::endl;
		std::cout << "\tCL_DEVICE_NAME        : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tCL_DEVICE_TYPE        : " << type << std::endl;
		std::cout << "\tApproach              : Naive" << std::endl;
		std::cout << "\tWindow size           : 5x5" << std::endl;
		std::cout << "\tResult		      : " << *resultsN << std::endl;
		std::cout << "\tAverage Execution Time: " << timeTotal / NUM_ITERATIONS << '\n' << std::endl;

		// ========================================= 7x7 Naive approach ===================================================
		std::cout << "===============7x7 Naive Approach===============\n" << std::endl;

		// set kernel arguments
		std::cout << "Setting kernel arguments for 7x7 image convolution" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsWBuffer);
		size = 7; // window 7x7
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		std::cout << "\tReading the result from the 7x7 Naive approach..." << std::endl;
		queue.enqueueReadBuffer(resultsWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &resultsN[0]);

		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		std::cout << "\tWriting output image '" << outputN7file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(outputN7file, outputImage, imgWidth, imgHeight);

		std::cout << "'" << outputN7file << "' is successfully created" << std::endl;

		// Compare the kernel execution times based on the different window sizes, 
		// and whether it was run on a multicore CPU or GPU
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Profiling Kernel Execution Time" << std::endl;
		std::cout << "\tCL_DEVICE_NAME        : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tCL_DEVICE_TYPE        : " << type << std::endl;
		std::cout << "\tApproach              : Naive" << std::endl;
		std::cout << "\tWindow size           : 7x7" << std::endl;
		std::cout << "\tResult		      : " << *resultsN << std::endl;
		std::cout << "\tAverage Execution Time: " << timeTotal / NUM_ITERATIONS << '\n' << std::endl;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// ====================================== 3x3 Horizontal Pass =================================================
		std::cout << "==============3x3 Horizontal Pass==============\n" << std::endl;
		// The first pass will be in the horizontal direction
		// create a kernel
		std::cout << "Creating a kernel to run filter in 3x3 horizontal pass" << std::endl;
		kernel = cl::Kernel(program, "horizontal_pass");

		// set kernel arguments
		std::cout << "Setting kernel arguments for 3x3 horizontal filter" << std::endl;
		kernel.setArg (0, inputImgBuffer);
		kernel.setArg (1, outputImgBuffer);
		kernel.setArg (2, resultsWBuffer);
		size = 3; // window 3x3
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		// The result will then undergo a second pass in the vertical direction
		// (enqueue the kernel twice to perform the blur in different directions)
		std::cout << "\tReading the result from the 3x3 first horizontal pass..."<< std::endl;
		queue.enqueueReadBuffer(resultsWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &results2P[0]);
		*tempResultsH = *results2P;

		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);
		
		std::cout << "\tWriting output image '" << output2PH3x3file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(output2PH3x3file, outputImage, imgWidth, imgHeight);
		
		std::cout << "'" << output2PH3x3file << "' is successfully created\n" << std::endl;
		
		// ===================================== 3x3 Vertical Pass ====================================================
		std::cout << "===============3x3 Vertical Pass===============\n" << std::endl;
		// The second pass will be in the vertical direction
		// create a kernel
		std::cout << "Creating a kernel to run filter in 3x3 vertical pass" << std::endl;
		kernel = cl::Kernel (program, "vertical_pass");

		// The result will then undergo a second pass in the vertical direction
		std::cout << "Updating buffer with filtered image from 3x3 horizontal pass" << std::endl;
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		resultsRWBuffer = cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 4, &results2P[0]);
		
		// set kernel arguments
		std::cout << "Setting kernel arguments for 3x3 vertical filter" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsRWBuffer);
		size = 3; // window 3x3
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		std::cout << "\tReading the result from the second 3x3 vertical pass..." << std::endl;
		queue.enqueueReadBuffer(resultsRWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &results2P[0]);
		*tempResultsV = *results2P;

		// enqueue command to read image from device to host memory
		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage (outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);
		
		// output results to image file
		std::cout << "\tWriting output image '" << output2P3x3file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB (output2P3x3file, outputImage, imgWidth, imgHeight);
		
		std::cout << "'" << output2P3x3file << "' is successfully created\n" << std::endl;

		// Compare the kernel execution times based on the different window sizes, 
		// and whether it was run on a multicore CPU or GPU
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Profiling Kernel Execution Time" << std::endl;
		std::cout << "\tCL_DEVICE_NAME        : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tCL_DEVICE_TYPE        : " << type << std::endl;
		std::cout << "\tApproach              : 2-pass" << std::endl;
		std::cout << "\tWindow size           : 3x3" << std::endl;
		std::cout << "\tResult                : " << *tempResultsV * *tempResultsH << std::endl;
		std::cout << "\tAverage Execution Time: " << timeTotal / (NUM_ITERATIONS*2) << '\n' << std::endl;

		// ====================================== 5x5 Horizontal Pass =================================================
		std::cout << "==============5x5 Horizontal Pass==============\n" << std::endl;
		// The first pass will be in the horizontal direction
		// create a kernel
		std::cout << "Creating a kernel to run filter in 5x5 horizontal pass" << std::endl;
		kernel = cl::Kernel(program, "horizontal_pass");

		// set kernel arguments
		std::cout << "Setting kernel arguments for 5x5 horizontal filter" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsWBuffer);
		size = 5; // window 5x5
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		// The result will then undergo a second pass in the vertical direction
		// (enqueue the kernel twice to perform the blur in different directions)
		std::cout << "\tReading the result from the 5x5 first horizontal pass..." << std::endl;
		queue.enqueueReadBuffer(resultsWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &results2P[0]);
		*tempResultsH = *results2P;

		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);
		
		std::cout << "\tWriting output image '" << output2PH5x5file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(output2PH5x5file, outputImage, imgWidth, imgHeight);

		std::cout << "'" << output2PH5x5file << "' is successfully created\n" << std::endl;

		// ===================================== 5x5 Vertical Pass ====================================================
		std::cout << "===============5x5 Vertical Pass===============\n" << std::endl;
		// The second pass will be in the vertical direction
		// create a kernel
		std::cout << "Creating a kernel to run filter in 5x5 vertical pass" << std::endl;
		kernel = cl::Kernel(program, "vertical_pass");

		// The result will then undergo a second pass in the vertical direction
		std::cout << "Updating buffer with filtered image from 5x5 horizontal pass" << std::endl;
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		resultsRWBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 4, &results2P[0]);

		// set kernel arguments
		std::cout << "Setting kernel arguments for 5x5 vertical filter" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsRWBuffer);
		size = 5; // window 5x5
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		std::cout << "\tReading the result from the second 5x5 vertical pass..." << std::endl;
		queue.enqueueReadBuffer(resultsRWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &results2P[0]);
		*tempResultsV = *results2P;

		// enqueue command to read image from device to host memory
		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		std::cout << "\tWriting output image '" << output2P5x5file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(output2P5x5file, outputImage, imgWidth, imgHeight);

		std::cout << "'" << output2P5x5file << "' is successfully created\n" << std::endl;

		// Compare the kernel execution times based on the different window sizes, 
		// and whether it was run on a multicore CPU or GPU
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Profiling Kernel Execution Time" << std::endl;
		std::cout << "\tCL_DEVICE_NAME        : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tCL_DEVICE_TYPE        : " << type << std::endl;
		std::cout << "\tApproach              : 2-pass" << std::endl;
		std::cout << "\tWindow size           : 5x5" << std::endl;
		std::cout << "\tResult                : " << *tempResultsV * *tempResultsH << std::endl;
		std::cout << "\tAverage Execution Time: " << timeTotal / (NUM_ITERATIONS * 2) << '\n' << std::endl;

		// ====================================== 7x7 Horizontal Pass =================================================
		std::cout << "==============7x7 Horizontal Pass==============\n" << std::endl;
		// The first pass will be in the horizontal direction
		// create a kernel
		std::cout << "Creating a kernel to run filter in 7x7 horizontal pass" << std::endl;
		kernel = cl::Kernel(program, "horizontal_pass");

		// set kernel arguments
		std::cout << "Setting kernel arguments for 7x7 horizontal filter" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsWBuffer);
		size = 7; // window 7x7
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		// The result will then undergo a second pass in the vertical direction
		// (enqueue the kernel twice to perform the blur in different directions)
		std::cout << "\tReading the result from the 7x7 first horizontal pass..." << std::endl;
		queue.enqueueReadBuffer(resultsWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &results2P[0]);
		*tempResultsH = *results2P;

		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		std::cout << "\tWriting output image '" << output2PH7x7file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(output2PH7x7file, outputImage, imgWidth, imgHeight);

		std::cout << "'" << output2PH7x7file << "' is successfully created\n" << std::endl;

		// ===================================== 7x7 Vertical Pass ====================================================
		std::cout << "===============7x7 Vertical Pass===============\n" << std::endl;
		// The second pass will be in the vertical direction
		// create a kernel
		std::cout << "Creating a kernel to run filter in 7x7 vertical pass" << std::endl;
		kernel = cl::Kernel(program, "vertical_pass");

		// The result will then undergo a second pass in the vertical direction
		std::cout << "Updating buffer with filtered image from 7x7 horizontal pass" << std::endl;
		inputImgBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imgWidth, imgHeight, 0, (void*)outputImage);
		resultsRWBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 4, &results2P[0]);

		// set kernel arguments
		std::cout << "Setting kernel arguments for 7x7 vertical filter" << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsRWBuffer);
		size = 7; // window 7x7
		wsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * 4, &size);
		kernel.setArg(3, wsizeBuffer);

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

		std::cout << "\tReading the result from the second 7x7 vertical pass..." << std::endl;
		queue.enqueueReadBuffer(resultsRWBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &results2P[0]);
		*tempResultsV = *results2P;

		// enqueue command to read image from device to host memory
		std::cout << "\tReading the results of a filtered image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		// output results to image file
		std::cout << "\tWriting output image '" << output2P7x7file << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(output2P7x7file, outputImage, imgWidth, imgHeight);

		std::cout << "'" << output2P7x7file << "' is successfully created\n" << std::endl;

		// Compare the kernel execution times based on the different window sizes, 
		// and whether it was run on a multicore CPU or GPU
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Profiling Kernel Execution Time" << std::endl;
		std::cout << "\tCL_DEVICE_NAME        : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tCL_DEVICE_TYPE        : " << type << std::endl;
		std::cout << "\tApproach              : 2-pass" << std::endl;
		std::cout << "\tWindow size           : 7x7" << std::endl;
		std::cout << "\tResult                : " << *tempResultsV * *tempResultsH << std::endl;
		std::cout << "\tAverage Execution Time: " << timeTotal / (NUM_ITERATIONS * 2) << '\n' << std::endl;
	
		// deallocate memory
		free (inputImage);
		free (outputImage);
	}
	catch (cl::Error e)
	{
		handle_error(e);
	}

	system ("pause");

	return 0;
}