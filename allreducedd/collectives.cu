#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <mpi.h>

#include "collectives.h"

struct MPIGlobalState {
    // The CUDA device to run on, or -1 for CPU-only.
    int device = -1;

    // A CUDA stream (if device >= 0) initialized on the device
    cudaStream_t stream;

    // Whether the global state (and MPI) has been initialized.
    bool initialized = false;
};

typedef struct doubledouble{
	double hi;
	double lo;
}dd_real;


// MPI relies on global state for most of its internal operations, so we cannot
// design a library that avoids global state. Instead, we centralize it in this
// single global struct.
static MPIGlobalState global_state;

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
//
// An exception is thrown if MPI or CUDA cannot be initialized.
void InitCollectives(int device) {
    if(device < 0) {
        // CPU-only initialization.
        int mpi_error = MPI_Init(NULL, NULL);
        if(mpi_error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init failed with an error");
        }

        global_state.device = -1;
    } else {
        // GPU initialization on the given device.
        //
        // For CUDA-aware MPI implementations, cudaSetDevice must be called
        // before MPI_Init is called, because MPI_Init will pick up the created
        // CUDA context and use it to create its own internal streams. It uses
        // these internal streams for data transfers, which allows it to
        // implement asynchronous sends and receives and allows it to overlap
        // GPU data transfers with whatever other computation the GPU may be
        // doing.
        //
        // It is not possible to control which streams the MPI implementation
        // uses for its data transfer.
        cudaError_t error = cudaSetDevice(device);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed with an error");
        }

        // When doing a CUDA-aware allreduce, the reduction itself (the
        // summation) must be done on the GPU with an elementwise arithmetic
        // kernel. We create our own stream to launch these kernels on, so that
        // the kernels can run independently of any other computation being done
        // on the GPU.
        cudaStreamCreate(&global_state.stream);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed with an error");
        }

        int mpi_error = MPI_Init(NULL, NULL);
        if(mpi_error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init failed with an error");
        }

        global_state.device = device;
    }
    global_state.initialized = true;
}

// Allocate a new memory buffer on CPU or GPU.
double* alloc(size_t size) {
    if(global_state.device < 0) {
        // CPU memory allocation through standard allocator.
        return new double[size];
    } else {
        // GPU memory allocation through CUDA allocator.
        void* memory;
        cudaError_t error = cudaMalloc(&memory, sizeof(double) * size);
        if(error != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed with an error");
        }
        return (double*) memory;
    }
}

// Deallocate an allocated memory buffer.
void dealloc(double* buffer) {
    if(global_state.device < 0) {
        // CPU memory deallocation through standard allocator.
        delete[] buffer;
    } else {
        // GPU memory deallocation through CUDA allocator.
        cudaFree(buffer);
    }
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(double* dst, double* src, size_t size) {
    if(global_state.device < 0) {
        // CPU memory allocation through standard allocator.
        std::memcpy((void*) dst, (void*) src, size * sizeof(double));
    } else {
        // GPU memory allocation through CUDA allocator.
        cudaMemcpyAsync((void*) dst, (void*) src, size * sizeof(double),
                        cudaMemcpyDeviceToDevice, global_state.stream);
        cudaStreamSynchronize(global_state.stream);
    }
}

// GPU kernel for adding two vectors elementwise.
__global__ void kernel_add(const double* x, const double* y, const int N, double* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
      out[i] = x[i] + y[i];
    }
}

double *TwoSum(double a,double b)
{
    double x,z,y;
    x=a+b;
    z=x-a;
    y=(a-(x-z))+(b-z);
    double *d=new double[2];
    d[0]=x;
    d[1]=y;
    return d;
}


dd_real quick_two_sum(double a,double b)
{
	dd_real res;
	res.hi=a+b;
	res.lo=b-(res.hi-a);
	return res;
}

dd_real two_sum(double a,double b)
{
	dd_real res;
	res.hi=a+b;
	double bb=res.hi-a;
	res.lo=(a-(res.hi-bb))+(b-bb);
	return res;
}

//reduce里面应该是调用这个算法，确保中间每个数都要用dd加，然后最后一个进程判断一下返回一个double再进行RingAllreduce
dd_real add_dd_dd(const dd_real aa,const dd_real bb)
{
	dd_real res,temp;
	temp=two_sum(aa.hi,bb.hi);
	temp.lo+=(aa.lo+bb.lo);
	res=quick_two_sum(temp.hi,temp.lo);
	return res;
}

double ddsum(dd_real *x,int size)
{
	int i;
	dd_real temp;
	double ddsumres;
	for(i=0;i<size-1;i++)
	{
		temp=add_dd_dd(x[i],x[i+1]);
		x[i+1]=temp;
	}
	ddsumres=x[size-1].hi+x[size-1].lo;
	return ddsumres;
}



// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
//two sum maybe can use double-double. or every time call sumk
//maybe need transform the error to next process
//可能需要每次把误差存起来，最后变成一个数组调用高精度求和算法
//得把一个chunk的数据都高精度加起来，而不仅仅是一个数，这样才是allreduce
//这个方法感觉会违背ring算法初衷
//这里的size参数应该是chunk大小
void reduce(dd_real* dst, dd_real* src, size_t size) {
    if(global_state.device < 0) {
        // Accumulate values from `src` into `dst` on the CPU.
	//double *temp=new double[2];
        for(size_t i = 0; i < size; i++) {
              //dst[i] += src[i];
//dddumv1
              dd_real temp;
	      temp=add_dd_dd(src[i],dst[i]);
	      dst[i]=temp;
//ddsum
//similar not effect
/*
	      dd_real *a=new dd_real[2];
	      //dd_real *a;
	      //a=(dd_real *)malloc(sizeof(dd_real)*2);
	      a[0].hi=src[i];
	      a[0].lo=0.00;
	      a[1].hi=dst[i];
	      a[1].lo=0.00;
	      dst[i]=ddsum(a,2);
*/	      
        }
    } else {
        // Launch a GPU kernel to accumulate values from `src` into `dst`.
	//报错在这行，我注释了
    //    kernel_add<<<32, 256, 0, global_state.stream>>>(src, dst, size, dst);
    //    cudaStreamSynchronize(global_state.stream);
    }
}

// Collect the input buffer sizes from all ranks using standard MPI collectives.
// These collectives are not as efficient as the ring collectives, but they
// transmit a very small amount of data, so that is OK.
std::vector<size_t> AllgatherInputLengths(int size, size_t this_rank_length) {
    std::vector<size_t> lengths(size);
    MPI_Allgather(&this_rank_length, 1, MPI_UNSIGNED_LONG,
                  &lengths[0], 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    return lengths;
}


/* Perform a ring allreduce on the data. The lengths of the data chunks passed
 * to this function must be the same across all MPI processes. The output
 * memory will be allocated and written into `output`.
 *
 * Assumes that all MPI processes are doing an allreduce of the same data,
 * with the same size.
 *
 * A ring allreduce is a bandwidth-optimal way to do an allreduce. To do the allreduce,
 * the nodes involved are arranged in a ring:
 *
 *                   .--0--.
 *                  /       \
 *                 3         1
 *                  \       /
 *                   *--2--*
 *
 *  Each node always sends to the next clockwise node in the ring, and receives
 *  from the previous one.
 *
 *  The allreduce is done in two parts: a scatter-reduce and an allgather. In
 *  the scatter reduce, a reduction is done, so that each node ends up with a
 *  chunk of the final output tensor which has contributions from all other
 *  nodes.  In the allgather, those chunks are distributed among all the nodes,
 *  so that all nodes have the entire output tensor.
 *
 *  Both of these operations are done by dividing the input tensor into N
 *  evenly sized chunks (where N is the number of nodes in the ring).
 *
 *  The scatter-reduce is done in N-1 steps. In the ith step, node j will send
 *  the (j - i)th chunk and receive the (j - i - 1)th chunk, adding it in to
 *  its existing data for that chunk. For example, in the first iteration with
 *  the ring depicted above, you will have the following transfers:
 *
 *      Segment 0:  Node 0 --> Node 1
 *      Segment 1:  Node 1 --> Node 2
 *      Segment 2:  Node 2 --> Node 3
 *      Segment 3:  Node 3 --> Node 0
 *
 *  In the second iteration, you'll have the following transfers:
 *
 *      Segment 0:  Node 1 --> Node 2
 *      Segment 1:  Node 2 --> Node 3
 *      Segment 2:  Node 3 --> Node 0
 *      Segment 3:  Node 0 --> Node 1
 *
 *  After this iteration, Node 2 has 3 of the four contributions to Segment 0.
 *  The last iteration has the following transfers:
 *
 *      Segment 0:  Node 2 --> Node 3
 *      Segment 1:  Node 3 --> Node 0
 *      Segment 2:  Node 0 --> Node 1
 *      Segment 3:  Node 1 --> Node 2
 *
 *  After this iteration, Node 3 has the fully accumulated Segment 0; Node 0
 *  has the fully accumulated Segment 1; and so on. The scatter-reduce is complete.
 *
 *  Next, the allgather distributes these fully accumululated chunks across all nodes.
 *  Communication proceeds in the same ring, once again in N-1 steps. At the ith step,
 *  node j will send chunk (j - i + 1) and receive chunk (j - i). For example, at the
 *  first iteration, the following transfers will occur:
 *
 *      Segment 0:  Node 3 --> Node 0
 *      Segment 1:  Node 0 --> Node 1
 *      Segment 2:  Node 1 --> Node 2
 *      Segment 3:  Node 2 --> Node 3
 *
 * After the first iteration, Node 0 will have a fully accumulated Segment 0
 * (from Node 3) and Segment 1. In the next iteration, Node 0 will send its
 * just-received Segment 0 onward to Node 1, and receive Segment 3 from Node 3.
 * After this has continued for N - 1 iterations, all nodes will have a the fully
 * accumulated tensor.
 *
 * Each node will do (N-1) sends for the scatter-reduce and (N-1) sends for the allgather.
 * Each send will contain K / N bytes, if there are K bytes in the original tensor on every node.
 * Thus, each node sends and receives 2K(N - 1)/N bytes of data, and the performance of the allreduce
 * (assuming no latency in connections) is constrained by the slowest interconnect between the nodes.
 *
 */
//感觉要把data改成dd
void RingAllreduce(double* data, size_t length, double** output_ptr) {
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Check that the lengths given to every process are the same.
    std::vector<size_t> lengths = AllgatherInputLengths(size, length);
    for(size_t other_length : lengths) {
        if(length != other_length) {
            throw std::runtime_error("RingAllreduce received different lengths");
        }
    }

    // Partition the elements of the array into N approximately equal-sized
    // chunks, where N is the MPI size.
    const size_t segment_size = length / size;
    std::vector<size_t> segment_sizes(size, segment_size);

    const size_t residual = length % size;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    // The last segment should end at the very end of the buffer.
    assert(segment_ends[size - 1] == length);

    // Allocate the output buffer.
    double* output = alloc(length);
    *output_ptr =  output;

    //dd_real **output_ptrdd;

    dd_real *outputdd;
    outputdd=(dd_real*)malloc(sizeof(dd_real)*(length));
    //*output_ptrdd=outputdd;

    dd_real *datadd;
    datadd=(dd_real*)malloc(sizeof(dd_real)*(length));
    for(int i=0;i<length;i++)
    {
	datadd[i].hi=data[i];
	datadd[i].lo=0;
	outputdd[i].hi=data[i];
	outputdd[i].lo=0;
    }

    // Copy your data to the output buffer to avoid modifying the input buffer.
    copy(output, data, length);

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.

    double* buffer = alloc(segment_sizes[0]);
    dd_real *bufferdd; 
    bufferdd=(dd_real*)malloc(sizeof(dd_real)*(segment_sizes[0]));

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % size;

    MPI_Status recv_status;
    MPI_Request recv_req;
    //MPI_Datatype datatype = MPI_DOUBLE;
    MPI_Datatype datatype=MPI_DOUBLE;
    MPI_Datatype ddtype;
    MPI_Type_contiguous(2,MPI_DOUBLE,&ddtype);
    MPI_Type_commit(&ddtype);

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, sends segment (rank - i) and receives
    // segment (rank - i - 1).
    //这里的size是进程总数
    //需要迭代size-1轮则每个进程有一个chunk有最终结果
    for (int i = 0; i < size - 1; i++) {
        int recv_chunk = (rank - i - 1 + size) % size;
        int send_chunk = (rank - i + size) % size;
        //double* segment_send = &(output[segment_ends[send_chunk] -
                                   //segment_sizes[send_chunk]]);
	dd_real* segment_senddd=&(outputdd[segment_ends[send_chunk]-segment_sizes[send_chunk]]);

        //MPI_Irecv(buffer, segment_sizes[recv_chunk],
                //datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);
	
	MPI_Irecv(bufferdd,segment_sizes[recv_chunk],ddtype,recv_from,0,MPI_COMM_WORLD,&recv_req);

        //MPI_Send(segment_send, segment_sizes[send_chunk],
                //MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD);

	MPI_Send(segment_senddd,segment_sizes[send_chunk],ddtype,send_to,0,MPI_COMM_WORLD);

        //double *segment_update = &(output[segment_ends[recv_chunk] -
          //                               segment_sizes[recv_chunk]]);

	dd_real *segment_updatedd=&(outputdd[segment_ends[recv_chunk]-segment_sizes[recv_chunk]]);

        // Wait for recv to complete before reduction
        MPI_Wait(&recv_req, &recv_status);

	//第一个参数是dst，第二个是src
        //dd_real *segment_updatedd;
	//dd_real *bufferdd;
	//segment_updatedd=(dd_real*)malloc(sizeof(dd_real)*(segement_sizes[recv_chunk]));
	//bufferdd=(dd_real*)malloc(sizeof(dd_real)*(segment_sizes[recv_chunk]));
/*
	for(int j=0;j<segment_sizes[recv_chunk];j++)
	{
	   segment_updatedd[j].hi=segment_update[j];
	   segment_updatedd[j].lo=0;
	   bufferdd[j].hi=buffer[j];
	   bufferdd[j].lo=0;	
	}
*/

        reduce(segment_updatedd, bufferdd, segment_sizes[recv_chunk]);
/*
	if(i==size-2)
	{
	    for(int k=0;k<segment_sizes[recv_chunk];k++)
	    {
		//这里应该是output,segment_update只是指向output地址
	        output[k]=segment_updatedd[k].hi+segment_updatedd[k].lo;
	    }
	}
*/
    }

    for(int k=0;k<length;k++)
    {
	output[k]=outputdd[k].hi+outputdd[k].lo;
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).
    for (size_t i = 0; i < size_t(size - 1); ++i) {
        int send_chunk = (rank - i + 1 + size) % size;
        int recv_chunk = (rank - i + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        double* segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        double* segment_recv = &(output[segment_ends[recv_chunk] -
                                       segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                datatype, send_to, 0, segment_recv,
                segment_sizes[recv_chunk], datatype, recv_from,
                0, MPI_COMM_WORLD, &recv_status);
    }

    // Free temporary memory.
    dealloc(buffer);
    MPI_Type_free(&ddtype);
}

// The ring allgather. The lengths of the data chunks passed to this function
// may differ across different devices. The output memory will be allocated and
// written into `output`.
//
// For more information on the ring allgather, read the documentation for the
// ring allreduce, which includes a ring allgather as the second stage.
/*
void RingAllgather(double* data, size_t length, double** output_ptr) {
    // Get MPI size and rank.
    int rank;
    int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    int size;
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(mpi_error != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    // Get the lengths of data provided to every process, so that we know how
    // much memory to allocate for the output buffer.
    std::vector<size_t> segment_sizes = AllgatherInputLengths(size, length);
    size_t total_length = 0;
    for(size_t other_length : segment_sizes) {
        total_length += other_length;
    }

    // Compute where each chunk ends.
    std::vector<size_t> segment_ends(size);
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < segment_ends.size(); ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    assert(segment_sizes[rank] == length);
    assert(segment_ends[size - 1] == total_length);

    // Allocate the output buffer and copy the input buffer to the right place
    // in the output buffer.
    double* output = alloc(total_length);
    *output_ptr = output;

    copy(output + segment_ends[rank] - segment_sizes[rank],
         data, segment_sizes[rank]);

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;
    MPI_Datatype datatype=MPI_DOUBLE;

    MPI_Status recv_status;

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).
    for (size_t i = 0; i < size_t(size - 1); ++i) {
        int send_chunk = (rank - i + size) % size;
        int recv_chunk = (rank - i - 1 + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        double* segment_send = &(output[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        double* segment_recv = &(output[segment_ends[recv_chunk] -
                                       segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                datatype, send_to, 0, segment_recv,
                segment_sizes[recv_chunk], datatype, recv_from,
                0, MPI_COMM_WORLD, &recv_status);
    }
}
*/
