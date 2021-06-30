/**
 * @file    parallel_heat_solver.h
 * @author  xlukac11 <xlukac11@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2020/2021 - Project 1
 *
 * @date    2021-4-27
 */

#ifndef PARALLEL_HEAT_SOLVER_H
#define PARALLEL_HEAT_SOLVER_H

#include "base_heat_solver.h"

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 1D and 2D block grid decomposition.
 */
class ParallelHeatSolver : public BaseHeatSolver
{
    //============================================================================//
    //                            *** BEGIN: NOTE ***
    //
    // Modify this class declaration as needed.
    // This class needs to provide at least:
    // - Constructor which passes SimulationProperties and MaterialProperties
    //   to the base class. (see below)
    // - Implementation of RunSolver method. (see below)
    // 
    // It is strongly encouraged to define methods and member variables to improve 
    // readability of your code!
    //
    //                             *** END: NOTE ***
    //============================================================================//
    
public:
    /**
     * @brief Constructor - Initializes the solver. This should include things like:
     *        - Construct 1D or 2D grid of tiles.
     *        - Create MPI datatypes used in the simulation.
     *        - Open SEQUENTIAL or PARALLEL HDF5 file.
     *        - Allocate data for local tile.
     *        - Initialize persistent communications?
     *
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    ParallelHeatSolver(SimulationProperties &simulationProps, MaterialProperties &materialProps);
    virtual ~ParallelHeatSolver();

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: The vector is allocated (and should be used) *ONLY*
     *                        by master process (rank 0 in MPI_COMM_WORLD)!
     */
    virtual void RunSolver(std::vector<float, AlignedAllocator<float> > &outResult);

    void StoreDataIntoFileParallel(hid_t fileHandle, size_t iteration, const float *data);

protected:

    // communicator for the reduce for the average temp computation
    MPI_Comm m_MIDDLE_COM;

    // TILE_BLOCK is created by spliting the actual grid
    MPI_Datatype m_MPI_TILE_BLOCK_FLOAT;
    MPI_Datatype m_MPI_TILE_BLOCK_INT;

    // TILE_TMP_BLOCK is the datatype for the tiles with borders with halo zones from the neighbors
    MPI_Datatype m_MPI_TILE_TMP_BLOCK_FLOAT;
    MPI_Datatype m_MPI_TILE_TMP_BLOCK_INT;

    // datatypes for the halo zones
    // RECV are datatypes that will recive data, SEND are halo zones that will be send to the neighbors
    MPI_Datatype m_RECV_FLOAT_LEFT, m_RECV_FLOAT_RIGHT, m_RECV_FLOAT_UP, m_RECV_FLOAT_DOWN;
    MPI_Datatype m_SEND_FLOAT_LEFT, m_SEND_FLOAT_RIGHT, m_SEND_FLOAT_UP, m_SEND_FLOAT_DOWN;

    // for this datatypes, each rank will make datatype based on the neighbor tile,
    // because thats where rank will store data when using RMA
    MPI_Datatype m_RMA_FLOAT_LEFT, m_RMA_FLOAT_RIGHT, m_RMA_FLOAT_UP, m_RMA_FLOAT_DOWN;

    // cartesian topology
    MPI_Comm m_TOPOLOGY;

    float m_airFlowRate;
    float m_coolerTemp;

    // windows for RMA
    MPI_Win m_win_1;
    MPI_Win m_win_2;

    // helper attributes for the calculation the average temp
    int m_middle_column;
    int m_middle_rank;

    int m_rank;     ///< Process rank in global (MPI_COMM_WORLD) communicator.
    int m_size;     ///< Total number of processes in MPI_COMM_WORLD.

    int m_tiles_x;  // number of tiles in x axis
    int m_tiles_y;  // number of tiles in y axis

    int m_idx_x;    // process tile in x axis
    int m_idx_y;    // process tile in y axis

    int m_tile_size_x;   // size of each process tile in x axis
    int m_tile_size_y;   // size of each process tile in y axis

    int m_tmp_tile_size_x;  // tmp tiles are tiles with margin for the 
    int m_tmp_tile_size_y;  // use with ComputePoint and also holding halo areas

    int left, right, top, down; // size of the margins
    
    AutoHandle<hid_t> m_fileHandle;
    
    std::vector<int, AlignedAllocator<int> > m_domain_map;
    std::vector<float, AlignedAllocator<float> > m_domain_params;
    std::vector<float, AlignedAllocator<float> > m_init_temp_1;
    std::vector<float, AlignedAllocator<float> > m_init_temp_2;
    std::vector<float, AlignedAllocator<float> > tile_parallel; // buffer tile for parallel I/O
    
    std::vector<int> m_element_count;       // element counts for the scatter and gather
    std::vector<int> m_element_displacement; // displacements of the TILE_BLOCK for the scatter and gather

    float ComputeMiddleColAvgTemp(const float *data) const;
};

#endif // PARALLEL_HEAT_SOLVER_H
