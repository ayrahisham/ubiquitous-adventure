// Name: Nur Suhaira Bte Badrul Hisham
// Student ID: 5841549
// Assignment 3 Part 2

// Without this, the compiler will use the default OpenCL 2.x, 
// where some OpenCL 1.2 functions are deprecated.
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

// By default, OpenCL disables exceptions.This line enables it.
#define __CL_ENABLE_EXCEPTIONS

// The headers from the C++ standard library and STL that are used
#include <iostream>
#include <fstream> // For file reading
#include <string>
#include <iomanip>

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
#define NUM_PIXELS_3 9
#define NUM_PIXELS_5 25
#define NUM_PIXELS_7 49

int main()
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::Kernel kernelReduction;
	cl::Kernel kernelComplete;
	cl::CommandQueue queue;			// commandqueue for a context and device

	// To perform profiling, an event must be associated with the command to profile
	cl::Event profileEvent;
	cl::Event startEvent;
	cl::Event endEvent;
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

	cl::Buffer sizeBuffer;			// image size buffer
	cl::Buffer resultsWBuffer;		// to store results (writing)
	cl::Buffer dataBuffer;			// to store pixel values of image
	cl::Buffer avgBuffer;			// to write average luminance calculated in kernel

	string type = "";

	cl::LocalSpaceArg localSpace;			// to create local space for the kernel
	size_t workgroupSize;					// work group size
	size_t kernelWorkgroupSize;				// allowed work group size for the kernel
	cl_ulong localMemorySize;				// device's local memory size
	std::vector<cl_float> vectorAvg;

	const char *outputfile = "outputGray.bmp"; // output image using grayscale conversion

	cl_int imagesize[2] = { 0 };
	cl_int pixelcount = 0; // number of pixels in image
	
	// average luminance for 2(b) in host
	cl_float avgH = 0.0;
	cl_float avgK = 0.0; // average luminance for 2(c) in kernel

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
		if (!build_program(&program, &context, "Part2.cl"))
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
		resultsWBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * (imgHeight * imgWidth));
		sizeBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * 2);

		std::cout << "===============Convert a colour image into a greyscale image===============\n" << std::endl;

		// create a kernel
		std::cout << "Creating a kernel for gray image conversion\n" << std::endl;
		kernel = cl::Kernel(program, "gray_scale");

		// set kernel arguments
		std::cout << "\tSetting kernel arguments for gray image conversion..." << std::endl;
		kernel.setArg(0, inputImgBuffer);
		kernel.setArg(1, outputImgBuffer);
		kernel.setArg(2, resultsWBuffer);
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

		std::cout << "\tReading the results of a gray image..." << std::endl;
		queue.enqueueReadImage(outputImgBuffer, CL_TRUE, origin, region, 0, 0, outputImage);

		std::cout << "\tWriting output image '" << outputfile << "' to RGB..." << std::endl;
		write_BMP_RGBA_to_RGB(outputfile, outputImage, imgWidth, imgHeight);

		std::cout << "'" << outputfile << "' is successfully created" << std::endl;

		// Compare the kernel execution times based on the different window sizes, 
		// and whether it was run on a multicore CPU or GPU
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Profiling Kernel Execution Time" << std::endl;
		std::cout << "\tCL_DEVICE_NAME        : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\tCL_DEVICE_TYPE        : " << type << std::endl;
		queue.enqueueReadBuffer(sizeBuffer, CL_TRUE, 0, sizeof(cl_int) * 2, &imagesize[0]);
		std::cout << "\tImage Width           : " << imagesize [0] << std::endl;
		std::cout << "\tImage Height          : " << imagesize [1] << std::endl;
		pixelcount = (imagesize[0] * imagesize[1]);
		std::cout << "\tAverage Execution Time: " << timeTotal / NUM_ITERATIONS << std::endl;

		// On the host, calculate the average luminance of the image by averaging the luminance
		// values of all pixels in the image.
		// (Note that if you use the example code from the tutorial on image processing, 
		// the R, G, and B values range from 
		// 0 to 255 on the host(unsigned char), and 0.0 to 1.0 (float)on the device.)
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Reading the result from color conversion" << std::endl;
		std::vector <cl_float> results (pixelcount);
		queue.enqueueReadBuffer(resultsWBuffer, CL_TRUE, 0, sizeof(cl_float) * pixelcount, &results[0]);
		std::cout << "Results from conversion: (from host)" << std::endl;
		// iterate through the imagesize
		for (int i = 0; i < pixelcount; i++)
		{
			// sum up all the pixel values
			avgH += results[i];
		}
		avgH *= 255.0;
		std::cout << "\tLuminance values of all pixels       : " << setprecision(2) << fixed << avgH << std::endl;
		std::cout << "\tNumber of pixels in '" << outputfile << "' : " << pixelcount << std::endl;
		/*(Note that if you use the example code from the tutorial on image processing,
		the R, G, and B values range from 0 to 255 on the host (unsigned char),
		and 0.0 to 1.0 (float) on the device.)*/
		avgH /= (pixelcount);
		std::cout << "\tAverage luminance of '" << outputfile << "': " << setprecision(2) << fixed << avgH << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;
		
		///////////////////////////////////////////////////////////////////////////////////////////////////////
		std::cout << "=============Parallel reduction to find the average luminance==============\n" << std::endl;

		// get device information and allowed kernel work group size
		workgroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		localMemorySize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
		kernelWorkgroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

		// display the information
		std::cout << "Checking sufficient local memory" << std::endl;
		std::cout << "Checking work-group size for kernel to run on\n";
		std::cout << device.getInfo<CL_DEVICE_NAME>() << ':' << std::endl;
		std::cout << "\tMaximum workgroup size: " << workgroupSize << std::endl;
		std::cout << "\tLocal memory size: " << localMemorySize << std::endl;
		std::cout << "\tKernel work-group size: " << kernelWorkgroupSize << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;

		// if kernel only allows one work-item per work-group, abort
		if (kernelWorkgroupSize == 1)
		{
			quit_program("Abort: Cannot run reduction kernel, because kernel workgroup size is 1.");
		}

		// if allowed kernel work group size smaller than device's max workgroup size
		if (workgroupSize > kernelWorkgroupSize)
		{
			workgroupSize = kernelWorkgroupSize;
		}

		// ensure sufficient local memory is available
		while (localMemorySize < sizeof(float) * workgroupSize * 4)
		{
			workgroupSize /= 4;
		}

		// create buffers
		dataBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (cl_float) * pixelcount, &results[0]);
		avgBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float));
		
		// set local space size
		localSpace = cl::Local(sizeof(float) * workgroupSize * 4);

		// create a kernel
		std::cout << "Creating a kernel for gray image conversion using parallel reduction\n" << std::endl;
		kernelReduction = cl::Kernel(program, "reduction_vector");
		kernelComplete = cl::Kernel(program, "reduction_complete");

		// set kernel arguments
		std::cout << "\tSetting kernel arguments for gray image conversion..." << std::endl;
		kernelReduction.setArg(0, dataBuffer);
		kernelReduction.setArg(1, localSpace);

		kernelComplete.setArg(0, dataBuffer);
		kernelComplete.setArg(1, localSpace);
		kernelComplete.setArg(2, avgBuffer);

		// enqueue kernel for execution
		size_t global = pixelcount / 4;

		cl::NDRange globalsize(global);
		cl::NDRange localSize(workgroupSize);

		queue.enqueueNDRangeKernel(kernelReduction, offset, globalsize, localSize, NULL, &startEvent);

		std::cout << "Global size: " << global << std::endl;

		// run reduction kernel until work-items can fit within one work-group
		while (global / workgroupSize > workgroupSize)
		{
			std::cout << "\tRun reduction kernel until work-items can fit within one work-group" << std::endl;
			global /= workgroupSize;
			globalsize = global;
			queue.enqueueNDRangeKernel(kernelReduction, offset, globalsize, localSize);
			std::cout << "\t\tGlobal size: " << global << std::endl;
		}

		// run reduction kernel one last time
		global /= workgroupSize;
		globalsize = global;
		queue.enqueueNDRangeKernel(kernelComplete, offset, globalsize, cl::NullRange, NULL, &endEvent);

		queue.finish();

		// check timing
		timeStart = startEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		timeEnd = endEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		timeTotal = timeEnd - timeStart;

		// read and check results
		queue.enqueueReadBuffer(avgBuffer, CL_TRUE, 0, sizeof(cl_float), &avgK);
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << "Execution time: " << timeTotal << std::endl;
		std::cout << "Results from conversion: (from opencl)" << std::endl;
		avgK *= 255.0;
		std::cout << "\tLuminance values of all pixels       : " << setprecision(2) << fixed << avgK << std::endl;
		std::cout << "\tNumber of pixels in '" << outputfile << "' : " << pixelcount << std::endl;
		/*(Note that if you use the example code from the tutorial on image processing,
		the R, G, and B values range from 0 to 255 on the host (unsigned char),
		and 0.0 to 1.0 (float) on the device.)*/
		avgK /= (pixelcount);
		std::cout << "\tAverage luminance of '" << outputfile << "': " << setprecision(2) << fixed << avgK << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;

		std::cout << "Checking result with host:" << std::endl;
		// slightly different results due to rounding errors
		if (fabs(avgK - avgH) > 1.0f)
		{
			std::cout << "\tCheck failed." << std::endl;
		}
		else
		{
			std::cout << "\tCheck passed." << std::endl;
		}

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