#include "collectives.h"
#include "timer.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>
#include <vector>
#include <math.h>

void TestCollectivesCPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Initialize on CPU (no GPU device ID).
    InitCollectives(NO_DEVICE);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    timer::Timer timer;
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];
	//printf("%d\n",size);
	//double need to change double
        double* data = new double[size];
        double seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            //这个有点问题，buffer_size为9，进程数为3，只有前3个数是3，应该是9个数都是3
            for(size_t j = 0; j < size; j++) {
		//data[j]=2.0f;
		//data[j]=j+1;
		data[j]=1.0f;
                //data[j] = sin(2*M_PI*(mpi_rank/(double)mpi_size-0.5));
		//std::cout<<data[j]<<" ";
		
            }
	    //printf("%.30e\n",data[0]);

            double* output = new double[size];
            timer.start();
            RingAllreduce(data, size, &output);
            seconds += timer.seconds();

            // Check that we get the expected result.
/*
            for(size_t j = 0; j < size; j++) {
                if(output[j] != (double) mpi_size) {
                    std::cerr << "Unexpected result from allreduce: " << data[j] << std::endl;
                    return;
                }
            }
*/
//            delete[] output;
//        }
        if(mpi_rank == 1) 
	{
	    for(size_t k=0;k<size;k++)
	    {
		//std::cout << output[k] <<" ";
		printf("%d process,ddringallreduce sum is:%.100e\n",mpi_size,output[k]);
		printf("time:%lf\n",seconds);
	    }
	    std::cout<<std::endl;   
/*
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds / iters
                << " per iteration)" << std::endl;
*/
        }
	
	delete[] output;
	}
        delete[] data;
    }
}

void TestCollectivesGPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Get the local rank, which gets us the GPU we should be using.
    //
    // We must do this before initializing MPI, because initializing MPI requires having the right
    // GPU context, so we use environment variables from our MPI implementation to determine the
    // local rank.
    // 
    // OpenMPI usually provides OMPI_COMM_WORLD_LOCAL_RANK, which we read. If you use SLURM with
    // OpenMPI, then SLURM instead provides SLURM_LOCALID. In this case, make sure to use `srun` or
    // `sbatch` and not `mpirun` to run your application.
    //
    // Remember that in order for this to work, you must have a GPU-enabled CUDA-aware MPI build.
    // Otherwise, this will result in a segfault, when MPI tries to read from a GPU memory pointer.
    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
        env_str = std::getenv("SLURM_LOCALID");
    }
    if(env_str == NULL) {
        throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK or SLURM_LOCALID!");
    }

    // Assume that the environment variable has an integer in it.
    int mpi_local_rank = std::stoi(std::string(env_str));
    InitCollectives(mpi_local_rank);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    cudaError_t err;

    timer::Timer timer;
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        double* cpu_data = new double[size];

        double* data;
        err = cudaMalloc(&data, sizeof(double) * size);
        if(err != cudaSuccess) { throw std::runtime_error("cudaMalloc failed with an error"); }

        double seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                cpu_data[j] = 1.0f;
            }

            err = cudaMemcpy(data, cpu_data, sizeof(double) * size, cudaMemcpyHostToDevice);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            double* output;
            timer.start();
            RingAllreduce(data, size, &output);
            seconds += timer.seconds();

            err = cudaMemcpy(cpu_data, output, sizeof(double) * size, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                if(cpu_data[j] != (double) mpi_size) {
                    std::cerr << "Unexpected result from allreduce: " << cpu_data[j] << std::endl;
                    return;
                }
            }
            err = cudaFree(output);
            if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        }
        if(mpi_rank == 0) {
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds / iters
                << " per iteration)" << std::endl;
        }

        err = cudaFree(data);
        if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        delete[] cpu_data;
    }
}

// Test program for baidu-allreduce collectives, should be run using `mpirun`.
int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: ./allreduce-test (cpu|gpu)" << std::endl;
        return 1;
    }
    std::string input(argv[1]);
	
    std::vector<size_t> buffer_sizes={9};
    std::vector<size_t> iterations={1};
    // Buffer sizes used for tests.
    /*
    std::vector<size_t> buffer_sizes = {
        0, 32, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 8388608, 67108864, 536870912
    };
*/
    // Number of iterations to run for each buffer size.
    /*
    std::vector<size_t> iterations = {
        100000, 100000, 100000, 100000,
        1000, 1000, 1000, 1000,
        100, 50, 10, 1
    };
*/

    // Test on either CPU and GPU.
    if(input == "cpu") {
        TestCollectivesCPU(buffer_sizes, iterations);
    } else if(input == "gpu") {
        TestCollectivesGPU(buffer_sizes, iterations);
    } else {
        std::cerr << "Unknown device type: " << input << std::endl
                  << "Usage: ./allreduce-test (cpu|gpu)" << std::endl;
        return 1;
    }

    // Finalize to avoid any MPI errors on shutdown.
    MPI_Finalize();

    return 0;
}
