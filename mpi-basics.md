# MPI

## *Message Passing Interface*
- standard for machine passing computation (been around for a long time)
- cross platform(desktop to supercomputer)
- language agnostic(C, C++, FOTRAN)

On a distributed machine, we have distributed pairs of cpu's and memories (Independent from one another) and the only way we can move information through them is through communication via a network, MPI basically supports this.

## Implementations
There a couple different implementations of MPI mainly through academia and industry collaborations.
- [MPICH](https://www.mpich.org/)
    - open source
    - 'CH' comes from the Chameleon portability system
    - various versions such as MPICH1, MPICH2, MPICH3
- [OpenMPI](https://www.open-mpi.org/)
    - open source
    - Collaborators include Amazon, IBM, Los Alamos Lab etc.

## Communication Domain
When we think about writing code in MPI, we're thinking about different processes which are running on different CPU's which will communicate with each other.
MPI communication patterns (which are supported by it) are abstracted out to something called as a 'Communicator'.
**Communicator type: MPI_Comm**
These are used in lot of the API calls.

## Common Functions
| **Function Purpose**                                 | **C Function Call**                                                                                               |
|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Initialize MPI                                       | int  MPI_Init(int *argc, char **argv)                                                                             |
| Determine number of processes within  a communicator | int  MPI_Comm_size(MPI_Comm comm, int *size)                                                                      |
| Determine processor rank within a communicator       | int  MPI_Comm_rank(MPI_Comm comm, int *rank)                                                                      |
| Exit MPI (must be called last by all processors)     | int  MPI_Finalize()                                                                                               |
| Send a message                                       | int MPI_Send (void *buf,int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)                       |
| Receive a message                                    | int MPI_Recv (void *buf,int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) |

These functions are very commonly used in MPI, the first of these (MPI_Init()) will configure our program to run MPI - initialize all of the data structures, process some command line calls etc. We need to call this function before we call any other commands.

At the end of our program we need to call - MPI_Finalize(). To be done before we stop the process, turn off any open channels etc.

After MPI_Init(), to understand the distributed process and its configuration we can run - MPI_Comm_size(), which gets us the number of processes communicating in our communicator.

Now, MPI_Comm_rank() tells us who we are! MPI creates its own global indexes for each of the participating processes.

**When we call MPI_Init() we pass our local memory address of argv and argc so that MPI can make any MPI related changes to our local variables argc and argv. MPI processes any MPI related arguments and removes them from the command line, so that our program doesn't have to configure itself for MPI related things.**

```
//Typical MPI initialization

int main(int argc, char **argv)
{   
    int num_procs;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); //Here we pass the reference of the variable num_procs to MPI, which will write a value to it
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);// Similarly, MPI will return a value for the rank of our process in the variable - rank.
    printf("%d: hello (p=%d)\n", rank, num_procs);

    /* Do many things, all at once */

    MPI_Finalize();
}

```
## Primitive Communication

Sending
```
/*Send*/
int MPI_Send(void *buf, int count, MPI_Daratype datatype, int dest, int tag, MPI_Comm comm)
```
Receiving 
```
/*Receive*/
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Commm comm, MPI_Status *status)
```
