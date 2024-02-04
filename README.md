# ddRingAllreduce
1.cd allreducedd
2.source /public/software/profile.d/mpi_openmpi-1.10.2-gnu.sh
（mpic++)
3.source /public/software/profile.d/cuda-7.5.sh
（cuda）
4.make MPI_ROOT=/public/software/mpi/openmpi/1.10.2/gnu CUDA_ROOT=/public/software/cuda-7.5
5.mpirun --np 3 allreduce-test cpu

We refer to
https://github.com/baidu-research/baidu-allreduce

our paper
Lei, X., Gu, T. & Xu, X. ddRingAllreduce: a high-precision RingAllreduce algorithm. CCF Trans. HPC 5, 245–257 (2023).
https://link.springer.com/article/10.1007/s42514-023-00150-2
