package opencl_examples;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_PLATFORM_NAME;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;
import static org.jocl.CL.clFinish;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

import opencl_examples.JOCLUtil;

public class MultipleQueue {

	/**
	 * The source code of the OpenCL program to execute
	 */
	private static final String source = "                                                    "
			+ "__kernel void arraySum(__global int *a){                                       "
			+ "    int gid = get_global_id(0);                                                "
			+ "    a[gid] = a[gid] + a[gid];                                                  "
			+ "}                                                                              "
			+ "                                                                               "
			+ "__kernel void arrayMultiply(__global int *a)                                   "
			+ "{                                                                              "
			+ "    int gid = get_global_id(0);                                                "
			+ "    a[gid] = a[gid] * a[gid];                                                  "
			+ "}                                                                              "
			+ "                                                                               "
			+ "__kernel void arrayIncrement(__global int *a) {                                "
			+ "    int gid = get_global_id(0);                                                "
			+ "    a[gid] = a[gid] + 1;                                                       "
			+ "}                                                                              ";

	/**
	 * No argument is required to run the kernels on multiple queues on the same
	 * context.
	 * 
	 * Good references related to OpenCL events
	 * (http://www.jocl.org/samples/JOCLEventSample.java)
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) {

		final int length = 1 << 24;// 2^24
		final int workgroupsize = 32;// the preferred group size multiple on my machine

		// Enable openCL exceptions
		CL.setExceptionsEnabled(true);
		//
		final int platformIndex = 0; // Platform index
		final long deviceType = CL.CL_DEVICE_TYPE_GPU; // Show GPU device type
		final int deviceIndex = 0; // Device number

		///////////////////////////// OpenCL setup code///////////////////////
		cl_platform_id[] platforms = JOCLUtil.getAllPlatforms();
		cl_platform_id platform = platforms[platformIndex];// Get the selected platform
		System.out.println("Selected CLPlatform: " + JOCLUtil.getPlatformInfoString(platform, CL_PLATFORM_NAME));// Show platform

		// Get all devices on the selected 'platform'
		cl_device_id[] devices = JOCLUtil.getAllDevices(platform, deviceType);
		cl_device_id device = devices[deviceIndex]; // Get a single device id
		System.out.println("Selected CLDevice: " + JOCLUtil.getDeviceInfoString(device, CL_DEVICE_NAME) + "\nDevice Version:"
				+ JOCLUtil.getDeviceInfoString(device, CL.CL_DEVICE_VERSION));// Show device

		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
		// Create a context for the selected device with contextProperties
		cl_context context = clCreateContext(contextProperties, 1, new cl_device_id[] { device }, null, null, null);
		/////////////////////////// End of OpenCL setup
		/////////////////////////// code///////////////////////////////////

		// create three queues on the same context of the GPU device.
		@SuppressWarnings("deprecation")
		cl_command_queue queue = clCreateCommandQueue(context, device, 0, null);
		@SuppressWarnings("deprecation")
		cl_command_queue queue2 = clCreateCommandQueue(context, device, 0, null);
		@SuppressWarnings("deprecation")
		cl_command_queue queue3 = clCreateCommandQueue(context, device, 0, null);
//		
		// Create input array 'a' and output array 'out'
		int a[] = new int[length];// Create array 'a' with given length
		int output[] = new int[length];
		// Populate array 'a' with 0 .. threads-1
		for (int i = 0; i < length; i++) {
			a[i] = i; // a[0] = 0,a[1] = 1, a[2] = 2, ....
		}
		// Allocate OpenCL-hosted memory for inputs and output
		Pointer ptrArray = Pointer.to(a);
		Pointer ptrOutput = Pointer.to(output);
		
		// Create a mutable (read-write) memory on OpenCL device
		// and copy the array 'a' from host to device
		// this memory object can be shared between queues created in the same context
		cl_mem memInOut = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * length,
				ptrArray, null);
		cl_program program = clCreateProgramWithSource(context, 1, new String[] { source }, null, null);
		clBuildProgram(program, 0, null, null, null, null);

		/////////////////////////////// Create three kernels
		cl_kernel sumkernel = clCreateKernel(program, "arraySum", null);
		clSetKernelArg(sumkernel, 0, Sizeof.cl_mem, Pointer.to(memInOut));
		cl_kernel multiplykernel = clCreateKernel(program, "arrayMultiply", null);
		clSetKernelArg(multiplykernel, 0, Sizeof.cl_mem, Pointer.to(memInOut));
		cl_kernel incrementKernel = clCreateKernel(program, "arrayIncrement", null);
		clSetKernelArg(incrementKernel, 0, Sizeof.cl_mem, Pointer.to(memInOut));
		// Set the work-item dimensions
		long global_work_size[] = new long[] { length }; // Global work group size is the number of repeats
		long local_work_size[] = new long[] { workgroupsize };

		final long time0 = System.nanoTime();
		/**
		 * Process tasks using in-order single queue
		 * 
		 * clEnqueueNDRangeKernel(queue, sumkernel, 1, null, global_work_size, local_work_size, 0, null, null);
		 *
		 * clEnqueueNDRangeKernel(queue, multiplykernel, 1, null, global_work_size, local_work_size, 0, null, null);
		 *
		 *	clEnqueueNDRangeKernel(queue, incrementKernel, 1, null, global_work_size, local_work_size, 0, null, null);
		 */

		 ////Process tasks using three separate queues and use cl_event to establish synchronization points 
		 ////'queue': 'sumkernel', queue2: 'multiplykernel' and queue3: 'incrementKernel'
		cl_event sumEvent = new cl_event();// sumEvent event is exposed by 'sumkernel' kernel
		clEnqueueNDRangeKernel(queue, sumkernel, 1, null, global_work_size, local_work_size, 0, null, sumEvent);

		cl_event multiplyEvent = new cl_event();// multiplyEvent event is exposed by 'multiplyKernel'
		clEnqueueNDRangeKernel(queue2, multiplykernel, 1, null, global_work_size, local_work_size, 0, null,
				multiplyEvent);
		cl_event incrementEvent = new cl_event();// incrementEvent event is exposed by 'incrementKernel'
		clEnqueueNDRangeKernel(queue3, incrementKernel, 1, null, global_work_size, local_work_size, 0, null,
				incrementEvent);
		// We need to specify the order of the kernels and wait for the kernels to complete.
		// Block everything until 'sum', 'multiply' and 'increment' kernels complete
		//try commenting out this line or changing the event order, and see what happens
		 CL.clWaitForEvents(3, new cl_event[] {sumEvent, multiplyEvent, incrementEvent });
	
		// Read all the results back to array 'output'		
		clEnqueueReadBuffer(queue, memInOut, CL_TRUE, 0, length * Sizeof.cl_int, ptrOutput, 0, null, null);
		
		//you can use WaitForEvent or clFinish to make sure clEnqueueReadBuffer command is complete before reading the data out.
		clFinish(queue);
		// Release memory objects, kernel, program, queue and context
		clReleaseMemObject(memInOut);
		clReleaseKernel(sumkernel);
		clReleaseKernel(multiplykernel);
		clReleaseKernel(incrementKernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseCommandQueue(queue2);
		clReleaseCommandQueue(queue3);
		clReleaseContext(context);

		final long time1 = System.nanoTime();
		System.out.println("Done in " + (time1 - time0) / 1000 + " microseconds");
//		//correct results 
//		//input:     [0] [1] [2]  [3]  [4] ....		
//		//add:       [0] [2] [4]  [6]  [8] ....
//		//multiply:  [0] [4] [16] [36] [64] ....
//		//increment: [1] [5] [17] [37] [65] [101] [145] [197] [257] [325] [401] [485] [577] [677] [785] [901] [1025] [1157] [1297] [1445] 
		for (int i = 0; i < 20; i++) {
			System.out.print("[" + output[i] + "] ");
		}

	}

}
