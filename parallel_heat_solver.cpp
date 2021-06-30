/**
 * @file    parallel_heat_solver.cpp
 * @author  xlukac11 <xlukac11@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2020/2021 - Project 1
 *
 * @date    2021-4-27
 */

#include "parallel_heat_solver.h"

using namespace std;

//============================================================================//
//                            *** BEGIN: NOTE ***
//
// Implement methods of your ParallelHeatSolver class here.
// Freely modify any existing code in ***THIS FILE*** as only stubs are provided 
// to allow code to compile.
//
//                             *** END: NOTE ***
//============================================================================//



ParallelHeatSolver::ParallelHeatSolver(SimulationProperties &simulationProps,
                                       MaterialProperties &materialProps)
    : BaseHeatSolver (simulationProps, materialProps),
    m_fileHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
{
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    
    // Creating EMPTY HDF5 handle using RAII "AutoHandle" type
    //
    // AutoHandle<hid_t> myHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
    //
    // This can be particularly useful when creating handle as class member!
    // Handle CANNOT be assigned using "=" or copy-constructor, yet it can be set
    // using ".Set(/* handle */, /* close/free function */)" as in:
    // myHandle.Set(H5Fopen(...), H5Fclose);
    
    // Requested domain decomposition can be queried by
    // m_simulationProperties.GetDecompGrid(/* TILES IN X */, /* TILES IN Y */)

    if(!m_simulationProperties.GetOutputFileName().empty()) {
        if ( m_rank == 0 && m_simulationProperties.IsUseParallelIO() == 0 ) {
            m_fileHandle.Set(H5Fcreate(m_simulationProperties.GetOutputFileName("par").c_str(),
                                    H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        }
        if ( m_simulationProperties.IsUseParallelIO() == 1 ) {
            hid_t accesPList = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(accesPList, MPI_COMM_WORLD, MPI_INFO_NULL);
            m_fileHandle.Set(H5Fcreate(m_simulationProperties.GetOutputFileName("par").c_str(),
                                    H5F_ACC_TRUNC, H5P_DEFAULT, accesPList), H5Fclose);
            H5Pclose(accesPList);
        }
    }
    
    // Get the grid decomposition
    m_simulationProperties.GetDecompGrid(m_tiles_x, m_tiles_y);

    // calculate position in the grid based on its decomposition
    // row-major pattern where x represents horizontal axis and y represents vertical axis
    m_idx_x = m_rank % m_tiles_x;
    m_idx_y = m_rank / m_tiles_x; // yes, m_tiles_x because we want next row in the matrix

    // get the size of the tile
    m_tile_size_x = m_materialProperties.GetEdgeSize() / m_tiles_x;
    m_tile_size_y = m_materialProperties.GetEdgeSize() / m_tiles_y;

    // create type for the tile for temperatures and params
    int og_array_size[] = {(int)m_materialProperties.GetEdgeSize(), (int)m_materialProperties.GetEdgeSize()};
    int tile_size[] = {m_tile_size_y, m_tile_size_x};
    int begin_0[] = {0, 0};

    // float type
    MPI_Datatype MPI_TILE_BLOCK_FLOAT;
    MPI_Type_create_subarray(2, // dimensions
                og_array_size,  // original array size
                tile_size,      // tile size
                begin_0,
                MPI_ORDER_C,
                MPI_FLOAT,
                &MPI_TILE_BLOCK_FLOAT);
    MPI_Type_commit(&MPI_TILE_BLOCK_FLOAT);
    MPI_Type_create_resized(MPI_TILE_BLOCK_FLOAT, 0, 1 * sizeof(float), &m_MPI_TILE_BLOCK_FLOAT);
    MPI_Type_commit(&m_MPI_TILE_BLOCK_FLOAT);

    // int type
    MPI_Datatype MPI_TILE_BLOCK_INT;
    MPI_Type_create_subarray(2, // dimensions
                og_array_size,  // original array size
                tile_size,      // tile size
                begin_0,
                MPI_ORDER_C,
                MPI_INT,
                &MPI_TILE_BLOCK_INT);
    MPI_Type_commit(&MPI_TILE_BLOCK_INT);
    MPI_Type_create_resized(MPI_TILE_BLOCK_INT, 0, 1 * sizeof(int), &m_MPI_TILE_BLOCK_INT);
    MPI_Type_commit(&m_MPI_TILE_BLOCK_INT);

    // margins for the tmp tiles
    m_idx_x == 0 ? left = 0 : left = 2;
    m_idx_x == m_tiles_x -1 ? right = 0 : right = 2;
    m_idx_y == 0 ? top = 0 : top = 2;
    m_idx_y == m_tiles_y -1 ? down = 0 : down = 2;
    
    // compute tmp tile size for the computation
    m_tmp_tile_size_x = m_tile_size_x + left + right;
    m_tmp_tile_size_y = m_tile_size_y + top + down;
    
    // create type for the tmp tile for temperatures and params
    int tmp_array_size[] = {m_tmp_tile_size_y, m_tmp_tile_size_x};
    int tile_start[] = {top, left}; // beginning of the tmp_tile in the in the tmp_block

    // float type
    MPI_Datatype MPI_TILE_TMP_BLOCK_FLOAT;
    MPI_Type_create_subarray(2,     // dimensions
                tmp_array_size,     // original array size
                tile_size,          // tile size
                tile_start,
                MPI_ORDER_C,
                MPI_FLOAT,
                &MPI_TILE_TMP_BLOCK_FLOAT);
    MPI_Type_commit(&MPI_TILE_TMP_BLOCK_FLOAT);
    MPI_Type_create_resized(MPI_TILE_TMP_BLOCK_FLOAT, 0, 1 * sizeof(float), &m_MPI_TILE_TMP_BLOCK_FLOAT);
    MPI_Type_commit(&m_MPI_TILE_TMP_BLOCK_FLOAT);
    
    // int type
    MPI_Datatype MPI_TILE_TMP_BLOCK_INT;
    MPI_Type_create_subarray(2,     // dimensions
                tmp_array_size,     // original array size
                tile_size,          // tile size
                tile_start,
                MPI_ORDER_C,
                MPI_INT,
                &MPI_TILE_TMP_BLOCK_INT);
    MPI_Type_commit(&MPI_TILE_TMP_BLOCK_INT);
    MPI_Type_create_resized(MPI_TILE_TMP_BLOCK_INT, 0, 1 * sizeof(int), &m_MPI_TILE_TMP_BLOCK_INT);
    MPI_Type_commit(&m_MPI_TILE_TMP_BLOCK_INT);

    // information for the scatterv,
    //m_element_displacement is the displacements of the tiles in whole matrix
    // m_element_count is all 1 because each rank has one tile
    m_element_count = std::vector<int>(m_size, 1);
    m_element_displacement = std::vector<int>(m_size, 0);
    for ( int i = 0; i < m_size; i++ ) {
        int tile_col = i % m_tiles_x;
        int tile_row = i / m_tiles_x;
        m_element_displacement[i] = m_tile_size_x * tile_col + m_materialProperties.GetEdgeSize() * m_tile_size_y * tile_row;
    }

    // scatter the initial temperature
    m_init_temp_1.resize(m_tmp_tile_size_x * m_tmp_tile_size_y);
    MPI_Scatterv(m_materialProperties.GetInitTemp().data(), //const void *sendbuf
                m_element_count.data(),                       //const int *sendcounts
                m_element_displacement.data(),                //const int *displs
                m_MPI_TILE_BLOCK_FLOAT,                     //MPI_Datatype sendtype
                m_init_temp_1.data(),                         //void *recvbuf,
                1,                                          //int recvcount,
                m_MPI_TILE_TMP_BLOCK_FLOAT,                 //MPI_Datatype recvtype
                0,                                          //int root
                MPI_COMM_WORLD);                            //MPI_Comm comm

    // resize m_init_temp_2 but scatter not needed as this is first new array
    m_init_temp_2.resize(m_tmp_tile_size_x * m_tmp_tile_size_y);
    std::copy(m_init_temp_1.begin(), m_init_temp_1.end(), m_init_temp_2.begin());

    // scatter the domain parameters
    m_domain_params.resize(m_tmp_tile_size_x * m_tmp_tile_size_y);
    MPI_Scatterv(m_materialProperties.GetDomainParams().data(),   //const void *sendbuf
                m_element_count.data(),                           //const int *sendcounts
                m_element_displacement.data(),                    //const int *displs
                m_MPI_TILE_BLOCK_FLOAT,                         //MPI_Datatype sendtype
                m_domain_params.data(),                         //void *recvbuf,
                1,                                              //int recvcount,
                m_MPI_TILE_TMP_BLOCK_FLOAT,                     //MPI_Datatype recvtype
                0,                                              //int root
                MPI_COMM_WORLD);                                //MPI_Comm comm

    // scatter the domain map
    m_domain_map.resize(m_tmp_tile_size_x * m_tmp_tile_size_y);
    MPI_Scatterv(m_materialProperties.GetDomainMap().data(), //const void *sendbuf
                m_element_count.data(),                        //const int *sendcounts
                m_element_displacement.data(),                 //const int *displs
                m_MPI_TILE_BLOCK_INT,                        //MPI_Datatype sendtype
                m_domain_map.data(),                         //void *recvbuf,
                1,                                           //int recvcount,
                m_MPI_TILE_TMP_BLOCK_INT,                    //MPI_Datatype recvtype
                0,                                           //int root
                MPI_COMM_WORLD);                             //MPI_Comm comm
    

    int rank_dims[] = {m_tiles_y, m_tiles_x};
    int periods[] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, // existující komunikátor
        2,                          // počet dimenzí
        rank_dims,                  // počet procesů v jednotlivých dims
        periods,                    // pole logických hodnot indikuje propojení do prstence když TRUE
        0,                          // FALSE=>indexy zachovány, TRUE=>možné přeindexování)
        &m_TOPOLOGY);               // nový kartézský komunikátor


    int column_size[] = {m_tile_size_y - (2-top) - (2-down), 2};
    int row_size[] = {2, m_tile_size_x - (2-left) - (2-right)};

    // sending zones
    MPI_Datatype SEND_INT_LEFT, SEND_INT_RIGHT, SEND_INT_UP, SEND_INT_DOWN;
    // zone positions
    int start_send_up[] = {2, 2};
    int start_send_down[] = {m_tmp_tile_size_y-4, 2};
    int start_send_left[] = {2, 2};
    int start_send_right[] = {2, m_tmp_tile_size_x-4};
    // create zones
    MPI_Type_create_subarray(2, tmp_array_size, row_size, start_send_up, MPI_ORDER_C, MPI_INT, &SEND_INT_UP);
    MPI_Type_create_subarray(2, tmp_array_size, row_size, start_send_down, MPI_ORDER_C, MPI_INT, &SEND_INT_DOWN);
    MPI_Type_create_subarray(2, tmp_array_size, column_size, start_send_left, MPI_ORDER_C, MPI_INT, &SEND_INT_LEFT);
    MPI_Type_create_subarray(2, tmp_array_size, column_size, start_send_right, MPI_ORDER_C, MPI_INT, &SEND_INT_RIGHT);
    // create zones
    MPI_Type_create_subarray(2, tmp_array_size, row_size, start_send_up, MPI_ORDER_C, MPI_FLOAT, &m_SEND_FLOAT_UP);
    MPI_Type_create_subarray(2, tmp_array_size, row_size, start_send_down, MPI_ORDER_C, MPI_FLOAT, &m_SEND_FLOAT_DOWN);
    MPI_Type_create_subarray(2, tmp_array_size, column_size, start_send_left, MPI_ORDER_C, MPI_FLOAT, &m_SEND_FLOAT_LEFT);
    MPI_Type_create_subarray(2, tmp_array_size, column_size, start_send_right, MPI_ORDER_C, MPI_FLOAT, &m_SEND_FLOAT_RIGHT);
    // commit
    MPI_Type_commit(&SEND_INT_UP);
    MPI_Type_commit(&SEND_INT_DOWN);
    MPI_Type_commit(&SEND_INT_LEFT);
    MPI_Type_commit(&SEND_INT_RIGHT);
    // commit
    MPI_Type_commit(&m_SEND_FLOAT_UP);
    MPI_Type_commit(&m_SEND_FLOAT_DOWN);
    MPI_Type_commit(&m_SEND_FLOAT_LEFT);
    MPI_Type_commit(&m_SEND_FLOAT_RIGHT);

    // recieving zones
    MPI_Datatype RECV_INT_LEFT, RECV_INT_RIGHT, RECV_INT_UP, RECV_INT_DOWN;
    // zone positions
    int start_recv_up[] = {0, 2};
    int start_recv_down[] = {m_tmp_tile_size_y-2, 2};
    int start_recv_left[] = {2, 0};
    int start_recv_right[] = {2, m_tmp_tile_size_x-2};
    // create zones
    MPI_Type_create_subarray(2, tmp_array_size, row_size, start_recv_up, MPI_ORDER_C, MPI_INT, &RECV_INT_UP);
    MPI_Type_create_subarray(2, tmp_array_size, row_size, start_recv_down, MPI_ORDER_C, MPI_INT, &RECV_INT_DOWN);
    MPI_Type_create_subarray(2, tmp_array_size, column_size, start_recv_left, MPI_ORDER_C, MPI_INT, &RECV_INT_LEFT);
    MPI_Type_create_subarray(2, tmp_array_size, column_size, start_recv_right, MPI_ORDER_C, MPI_INT, &RECV_INT_RIGHT);
    // create zones
    MPI_Type_create_subarray(2, tmp_array_size, row_size, start_recv_up, MPI_ORDER_C, MPI_FLOAT, &m_RECV_FLOAT_UP);
    MPI_Type_create_subarray(2, tmp_array_size, row_size, start_recv_down, MPI_ORDER_C, MPI_FLOAT, &m_RECV_FLOAT_DOWN);
    MPI_Type_create_subarray(2, tmp_array_size, column_size, start_recv_left, MPI_ORDER_C, MPI_FLOAT, &m_RECV_FLOAT_LEFT);
    MPI_Type_create_subarray(2, tmp_array_size, column_size, start_recv_right, MPI_ORDER_C, MPI_FLOAT, &m_RECV_FLOAT_RIGHT);
    // commit
    MPI_Type_commit(&RECV_INT_UP);
    MPI_Type_commit(&RECV_INT_DOWN);
    MPI_Type_commit(&RECV_INT_LEFT);
    MPI_Type_commit(&RECV_INT_RIGHT);
    // commit
    MPI_Type_commit(&m_RECV_FLOAT_UP);
    MPI_Type_commit(&m_RECV_FLOAT_DOWN);
    MPI_Type_commit(&m_RECV_FLOAT_LEFT);
    MPI_Type_commit(&m_RECV_FLOAT_RIGHT);

    if ( m_simulationProperties.IsRunParallelRMA() ) {

        // y_up, x_up, y_down, x_down, y_left, x_left, y_right, x_right
        int neighbor_array_tile_size[8];

        // neighbors exange size of their tiles
        MPI_Neighbor_allgather(tmp_array_size,          //const void *sendbuf,
                            2,                          //int sendcount,
                            MPI_INT,                    //MPI_Datatype sendtype,
                            neighbor_array_tile_size,   //void *recvbuf,
                            2,                          //int recvcount,
                            MPI_INT,                    //MPI_Datatype recvtype,
                            m_TOPOLOGY);                //MPI_Comm comm

        // datatype for sending up
        if ( (m_rank - m_tiles_x) >= 0 ) {
            int neighbor_tile_size[] = {neighbor_array_tile_size[0], neighbor_array_tile_size[1]};
            int neighbor_start[] = {neighbor_array_tile_size[0] - 2, 2};
            MPI_Type_create_subarray(2, neighbor_tile_size, row_size, neighbor_start,
                                        MPI_ORDER_C, MPI_FLOAT, &m_RMA_FLOAT_DOWN);
            MPI_Type_commit(&m_RMA_FLOAT_DOWN);
        }

        // datatype for sending down
        if ( (m_rank + m_tiles_x) < m_size ) {
            int neighbor_tile_size[] = {neighbor_array_tile_size[2], neighbor_array_tile_size[3]};
            int neighbor_start[] = {0, 2};
            MPI_Type_create_subarray(2, neighbor_tile_size, row_size, neighbor_start,
                                        MPI_ORDER_C, MPI_FLOAT, &m_RMA_FLOAT_UP);
            MPI_Type_commit(&m_RMA_FLOAT_UP);
        }

        // datatype for sending left
        if ( m_rank % m_tiles_x != 0 ) {
            int neighbor_tile_size[] = {neighbor_array_tile_size[4], neighbor_array_tile_size[5]};
            int neighbor_start[] = {2, neighbor_array_tile_size[5] - 2};
            MPI_Type_create_subarray(2, neighbor_tile_size, column_size, neighbor_start,
                                        MPI_ORDER_C, MPI_FLOAT, &m_RMA_FLOAT_RIGHT);
            MPI_Type_commit(&m_RMA_FLOAT_RIGHT);
        }

        // datatype for sending right
        if ( (m_rank + 1) % m_tiles_x != 0 ) {
            int neighbor_tile_size[] = {neighbor_array_tile_size[6], neighbor_array_tile_size[7]};
            int neighbor_start[] = {2, 0};
            MPI_Type_create_subarray(2, neighbor_tile_size, column_size, neighbor_start,
                                        MPI_ORDER_C, MPI_FLOAT, &m_RMA_FLOAT_LEFT);
            MPI_Type_commit(&m_RMA_FLOAT_LEFT);
        }

        MPI_Win_create(
            m_init_temp_1.data(),                   // Pointer na lokální data
            m_init_temp_1.size() * sizeof(float),   // Velikost lokání části okna v bytech
            sizeof(float),                          // Velikost jedné jednotky v bytech
            MPI_INFO_NULL,                          // Info objekt s optimalizacemi
            MPI_COMM_WORLD,                         // Komunikátor
            &m_win_1);                                // window

        MPI_Win_create(
            m_init_temp_2.data(),                   // Pointer na lokální data
            m_init_temp_2.size() * sizeof(float),   // Velikost lokání části okna v bytech
            sizeof(float),                          // Velikost jedné jednotky v bytech
            MPI_INFO_NULL,                          // Info objekt s optimalizacemi
            MPI_COMM_WORLD,                         // Komunikátor
            &m_win_2);                                // window
    }
    
    // send halo zones to the neighbors for the first time, the initial data
    int sendcounts[4] = {1, 1, 1, 1};
    int recvcounts[4] = {1, 1, 1, 1};
    MPI_Aint sdispls[4] = {0, 0, 0, 0};
    MPI_Aint rdispls[4] = {0, 0, 0, 0};
    MPI_Datatype sendtypes_d[4] = {SEND_INT_UP, SEND_INT_DOWN, SEND_INT_LEFT, SEND_INT_RIGHT};
    MPI_Datatype recvtypes_d[4] = {RECV_INT_UP, RECV_INT_DOWN, RECV_INT_LEFT, RECV_INT_RIGHT};
    MPI_Neighbor_alltoallw(m_domain_map.data(),     // const void *sendbuf
                    sendcounts,                     // const int sendcounts[]
                    sdispls,                        // const MPI_Aint sdispls[]
                    sendtypes_d,                    // const MPI_Datatype sendtypes[]
                    m_domain_map.data(),            // void *recvbuf
                    recvcounts,                     // const int recvcounts[]
                    rdispls,                        // const MPI_Aint rdispls[]
                    recvtypes_d,                    // const MPI_Datatype recvtypes[]
                    m_TOPOLOGY);                    // MPI_Comm comm
    
    MPI_Datatype sendtypes_t[4] = {m_SEND_FLOAT_UP, m_SEND_FLOAT_DOWN, m_SEND_FLOAT_LEFT, m_SEND_FLOAT_RIGHT};
    MPI_Datatype recvtypes_t[4] = {m_RECV_FLOAT_UP, m_RECV_FLOAT_DOWN, m_RECV_FLOAT_LEFT, m_RECV_FLOAT_RIGHT};
    MPI_Neighbor_alltoallw(m_init_temp_1.data(),    // const void *sendbuf
                    sendcounts,                     // const int sendcounts[]
                    sdispls,                        // const MPI_Aint sdispls[]
                    sendtypes_t,                    // const MPI_Datatype sendtypes[]
                    m_init_temp_1.data(),           // void *recvbuf
                    recvcounts,                     // const int recvcounts[]
                    rdispls,                        // const MPI_Aint rdispls[]
                    recvtypes_t,                    // const MPI_Datatype recvtypes[]
                    m_TOPOLOGY);                    // MPI_Comm comm*/
            
    MPI_Neighbor_alltoallw(m_domain_params.data(),    // const void *sendbuf
                    sendcounts,                     // const int sendcounts[]
                    sdispls,                        // const MPI_Aint sdispls[]
                    sendtypes_t,                    // const MPI_Datatype sendtypes[]
                    m_domain_params.data(),           // void *recvbuf
                    recvcounts,                     // const int recvcounts[]
                    rdispls,                        // const MPI_Aint rdispls[]
                    recvtypes_t,                    // const MPI_Datatype recvtypes[]
                    m_TOPOLOGY);                    // MPI_Comm comm*/


    // allocate buffer tile for parallel I/O
    if ( m_simulationProperties.IsUseParallelIO() == 1 )
        tile_parallel.resize(m_tile_size_y * m_tile_size_x);


    // create communicator for computing the middle average temperature
    // If the rank is in the first right tile then make com for it and set index of the center column
    if ( m_rank % m_tiles_x == m_tiles_x / 2 ) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, m_rank / m_tiles_y, &m_MIDDLE_COM);
        if ( m_tiles_x == 1 )
            m_middle_column = m_tmp_tile_size_x / 2;
        else
            m_middle_column = 2;
        MPI_Comm_rank(m_MIDDLE_COM, &m_middle_rank);
    }
    // Else, the rank is not in the middle com
    else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, m_rank, &m_MIDDLE_COM);
        m_middle_rank = -1;
    }

    m_airFlowRate = m_simulationProperties.GetAirFlowRate();
    m_coolerTemp = m_materialProperties.GetCoolerTemp();
}

ParallelHeatSolver::~ParallelHeatSolver() {

    MPI_Type_free(&m_MPI_TILE_BLOCK_FLOAT);
    MPI_Type_free(&m_MPI_TILE_BLOCK_INT);
    MPI_Type_free(&m_MPI_TILE_TMP_BLOCK_FLOAT);
    MPI_Type_free(&m_MPI_TILE_TMP_BLOCK_INT);

    MPI_Type_free(&m_RECV_FLOAT_LEFT);
    MPI_Type_free(&m_RECV_FLOAT_RIGHT);
    MPI_Type_free(&m_RECV_FLOAT_UP);
    MPI_Type_free(&m_RECV_FLOAT_DOWN);

    MPI_Type_free(&m_SEND_FLOAT_LEFT);
    MPI_Type_free(&m_SEND_FLOAT_RIGHT);
    MPI_Type_free(&m_SEND_FLOAT_UP);
    MPI_Type_free(&m_SEND_FLOAT_DOWN);

    if ( MPI_COMM_NULL != m_MIDDLE_COM )
        MPI_Comm_free(&m_MIDDLE_COM);
    MPI_Comm_free(&m_TOPOLOGY);

    if ( m_simulationProperties.IsRunParallelRMA() ) {
        if ( (m_rank - m_tiles_x) >= 0 ) 
            MPI_Type_free(&m_RMA_FLOAT_DOWN);

        if ( (m_rank + m_tiles_x) < m_size )
            MPI_Type_free(&m_RMA_FLOAT_UP);


        if ( m_rank % m_tiles_x != 0 )
            MPI_Type_free(&m_RMA_FLOAT_RIGHT);
        

        if ( (m_rank + 1) % m_tiles_x != 0 )
            MPI_Type_free(&m_RMA_FLOAT_LEFT);
        

        MPI_Win_free(&m_win_1);
        MPI_Win_free(&m_win_2);
    }
}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float> > &outResult)
{
    // UpdateTile(...) method can be used to evaluate heat equation over 2D tile
    //                 in parallel (using OpenMP).
    // NOTE: This method might be inefficient when used for small tiles such as 
    //       2xN or Nx2 (these might arise at edges of the tile)
    //       In this case ComputePoint may be called directly in loop.
    
    // ShouldPrintProgress(N) returns true if average temperature should be reported
    // by 0th process at Nth time step (using "PrintProgressReport(...)").
    
    // Finally "PrintFinalReport(...)" should be used to print final elapsed time and
    // average temperature in column.
    
    int sendcounts[4] = {1, 1, 1, 1};
    int recvcounts[4] = {1, 1, 1, 1};
    MPI_Aint sdispls[4] = {0, 0, 0, 0};
    MPI_Aint rdispls[4] = {0, 0, 0, 0};
    MPI_Datatype sendtypes[4] = {m_SEND_FLOAT_UP, m_SEND_FLOAT_DOWN, m_SEND_FLOAT_LEFT, m_SEND_FLOAT_RIGHT};
    MPI_Datatype recvtypes[4] = {m_RECV_FLOAT_UP, m_RECV_FLOAT_DOWN, m_RECV_FLOAT_LEFT, m_RECV_FLOAT_RIGHT};
    
    float *workTempArrays[] = { m_init_temp_2.data(), m_init_temp_1.data() };
    MPI_Win workWindows[] = { m_win_2, m_win_1 };
    float middleColAvgTemp = 0.0f;

    double startTime = MPI_Wtime();

    MPI_Request ineighbor_alltoallw_request;

    // Begin iterative simulation main loop
    for(size_t iter = 0; iter < m_simulationProperties.GetNumIterations(); ++iter)
    {
        // Compute new temperatures for the zones that will be exchanged
        // border temperatures should remain constant (plus our stencil is +/-2 points).
        // top tile
        for( int i = 2; i < 4; ++i ) {
            for( int j = 2; j < m_tmp_tile_size_x - 2; ++j )
                ComputePoint(workTempArrays[1], workTempArrays[0], m_domain_params.data(), m_domain_map.data(), i, j,
                    m_tmp_tile_size_x, m_airFlowRate, m_coolerTemp);
        }
        // bottom tile
        for( int i = m_tmp_tile_size_y - 4; i < m_tmp_tile_size_y - 2; ++i ) {
            for( int j = 2; j < m_tmp_tile_size_x - 2; ++j )
                ComputePoint(workTempArrays[1], workTempArrays[0], m_domain_params.data(), m_domain_map.data(), i, j,
                    m_tmp_tile_size_x, m_airFlowRate, m_coolerTemp);
        }
        // left tile
        for( int i = 4; i < m_tmp_tile_size_y - 4; ++i ) {
            for( int j = 2; j < 4; ++j )
                ComputePoint(workTempArrays[1], workTempArrays[0], m_domain_params.data(), m_domain_map.data(), i, j,
                    m_tmp_tile_size_x, m_airFlowRate, m_coolerTemp);
        }
        // right tile
        for( int i = 4; i < m_tmp_tile_size_y - 4; ++i ) {
            for( int j = m_tmp_tile_size_x - 4; j < m_tmp_tile_size_x - 2; ++j )
                ComputePoint(workTempArrays[1], workTempArrays[0], m_domain_params.data(), m_domain_map.data(), i, j,
                    m_tmp_tile_size_x, m_airFlowRate, m_coolerTemp);
        }
        
        // exchange halo areas using non-blocking collective comunication
        if ( !m_simulationProperties.IsRunParallelRMA() ) {
            MPI_Ineighbor_alltoallw(workTempArrays[0],  // const void *sendbuf
                    sendcounts,                         // const int sendcounts[]
                    sdispls,                            // const MPI_Aint sdispls[]
                    sendtypes,                          // const MPI_Datatype sendtypes[]
                    workTempArrays[0],                  // void *recvbuf
                    recvcounts,                         // const int recvcounts[]
                    rdispls,                            // const MPI_Aint rdispls[]
                    recvtypes,                          // const MPI_Datatype recvtypes[]
                    m_TOPOLOGY,                         // MPI_Comm comm
                    &ineighbor_alltoallw_request);      // MPI_Request * request
        } else {
            // make barier for the RMA halo zone exchange
            MPI_Win_fence(0, workWindows[0]);
            
            // send up
            if ( (m_rank - m_tiles_x) >= 0 )
                MPI_Put(
                    workTempArrays[0],  // Odkud
                    1,                  // Kolik prvků
                    m_SEND_FLOAT_UP,    // Jakého datového typy
                    m_rank - m_tiles_x, // Ke komu zapsat
                    0,                  // Na jaké místo (od kolikátého prvku v okně)
                    1,                  // Kolik prvků
                    m_RMA_FLOAT_DOWN,   // Jakého datového typu
                    workWindows[0]);
            
            // send down
            if ( (m_rank + m_tiles_x) < m_size )
                MPI_Put(workTempArrays[0], 1, m_SEND_FLOAT_DOWN, m_rank + m_tiles_x, 0, 1, m_RMA_FLOAT_UP, workWindows[0]);

            // send left
            if ( m_rank % m_tiles_x != 0 )
                MPI_Put(workTempArrays[0], 1, m_SEND_FLOAT_LEFT, m_rank - 1, 0, 1, m_RMA_FLOAT_RIGHT, workWindows[0]);

            // send right
            if ( (m_rank + 1) % m_tiles_x != 0 )
                MPI_Put(workTempArrays[0], 1, m_SEND_FLOAT_RIGHT, m_rank + 1, 0, 1, m_RMA_FLOAT_LEFT, workWindows[0]);

        }

        // continue calculating the remaining data in tile
        for( int i = 4; i < m_tmp_tile_size_y - 4; ++i ) {
            for( int j = 4; j < m_tmp_tile_size_x - 4; ++j ) {
                ComputePoint(workTempArrays[1],                 // old temp
                        workTempArrays[0],                      // new temp
                        m_domain_params.data(),                 // params
                        m_domain_map.data(),                    // map
                        i,                                      // row
                        j,                                      // col
                        m_tmp_tile_size_x,                      // width
                        m_airFlowRate,                          // airflow
                        m_coolerTemp);                          // cooler temp
            }
        }

        // Store the simulation state if appropriate (ie. every N-th iteration)
        if ( !m_simulationProperties.GetOutputFileName().empty() &&
            ((iter % m_simulationProperties.GetDiskWriteIntensity()) == 0) ) {

            // Do gather to get all tiles if non-parallel I/O
            if ( m_simulationProperties.IsUseParallelIO() == 0 )
                MPI_Gatherv(workTempArrays[0],                      // const void *sendbuf
                    1,                                          // int sendcount
                    m_MPI_TILE_TMP_BLOCK_FLOAT,                 // MPI_Datatype sendtype
                    m_materialProperties.GetInitTemp().data(),  // void *recvbuf
                    m_element_count.data(),                     // const int *recvcounts
                    m_element_displacement.data(),              // const int *displs
                    m_MPI_TILE_BLOCK_FLOAT,                     // MPI_Datatype recvtype
                    0,                                          // int root
                    MPI_COMM_WORLD);                            // MPI_Comm comm
                    
            if ( m_fileHandle != H5I_INVALID_HID ) {

                // root rank saves the simulation if it's non-parallel I/O
                if ( m_rank == 0 && m_simulationProperties.IsUseParallelIO() == 0 )
                    StoreDataIntoFile(m_fileHandle, iter, m_materialProperties.GetInitTemp().data());

                // every rank saves the simulation if its parallel I/O
                if ( m_simulationProperties.IsUseParallelIO() == 1 ){
                    
                    // every rank sends the data from tmp tile to the actual tile for itself
                    MPI_Request req;
                    MPI_Isend(workTempArrays[0], 1, m_MPI_TILE_TMP_BLOCK_FLOAT, 0, 0, MPI_COMM_SELF, &req);
                    MPI_Recv(tile_parallel.data(), m_tile_size_y * m_tile_size_x, MPI_FLOAT, 0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE);
                    MPI_Wait(&req, MPI_STATUS_IGNORE);

                    StoreDataIntoFileParallel(m_fileHandle, iter, tile_parallel.data());
                }
            }
        }

        // Compute average temperature in the middle column of the domain.
        middleColAvgTemp = ComputeMiddleColAvgTemp(workTempArrays[0]);

        // Print current progress (prints progress only every 10% of the simulation).
        if ( m_middle_rank == 0 || m_rank == 0 ) {
            if ( m_rank != m_middle_rank ) {
                if ( m_middle_rank == 0 )
                    MPI_Send(&middleColAvgTemp, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
                else{
                    MPI_Recv(&middleColAvgTemp, 1, MPI_FLOAT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    PrintProgressReport(iter, middleColAvgTemp);
                }
            } else
                PrintProgressReport(iter, middleColAvgTemp);
        }
        
        // wait for the request from non-blocking neighbor alltoallw or use fence for the RMA
        if ( !m_simulationProperties.IsRunParallelRMA() ) {
            MPI_Wait(&ineighbor_alltoallw_request, MPI_STATUS_IGNORE);
        } else {
            MPI_Win_fence(0, workWindows[0]);
        }

        // 7. Swap source and destination buffers
        std::swap(workTempArrays[0], workTempArrays[1]);
        std::swap(workWindows[0], workWindows[1]);
    }

    // Measure total execution time and report
    if ( m_rank == 0 ){
        double elapsedTime = MPI_Wtime() - startTime;
        PrintFinalReport(elapsedTime, middleColAvgTemp, "par");
    }
    
    // Gather the last iteration outputs and copy them into output buffer
    MPI_Gatherv(workTempArrays[1],                  // const void *sendbuf
                    1,                              // int sendcount
                    m_MPI_TILE_TMP_BLOCK_FLOAT,     // MPI_Datatype sendtype
                    outResult.data(),               // void *recvbuf
                    m_element_count.data(),         // const int *recvcounts
                    m_element_displacement.data(),  // const int *displs
                    m_MPI_TILE_BLOCK_FLOAT,         // MPI_Datatype recvtype
                    0,                              // int root
                    MPI_COMM_WORLD);

}

float ParallelHeatSolver::ComputeMiddleColAvgTemp(const float *data) const {

    if ( MPI_COMM_NULL == m_MIDDLE_COM )
        return 0.0;
    
    float middleColAvgTemp = 0.0f;
    for(int i = top; i < m_tmp_tile_size_y - down; ++i)
        middleColAvgTemp += data[i*m_tmp_tile_size_x + m_middle_column];

    float res = 0;
    MPI_Reduce(&middleColAvgTemp,
        &res,               // Význam pouze u root
        1,                  // Počet operací na prvcích sendbuf
        MPI_FLOAT,
        MPI_SUM,            // asociativní operátor
        0,                  // kdo získá výsledek
        m_MIDDLE_COM);
    
    return res / float(m_materialProperties.GetEdgeSize());
}

void ParallelHeatSolver::StoreDataIntoFileParallel(hid_t fileHandle, size_t iteration,
                                       const float *data)
{
    hsize_t gridSize[] = { m_materialProperties.GetEdgeSize(), m_materialProperties.GetEdgeSize() };

    // 1. Create new HDF5 file group named as "Timestep_N", where "N" is number
    //    of current snapshot. The group is placed into root of the file "/Timestep_N".
    std::string groupName = "Timestep_" + std::to_string(static_cast<unsigned long long>(iteration / m_simulationProperties.GetDiskWriteIntensity()));
    AutoHandle<hid_t> groupHandle(H5Gcreate(fileHandle, groupName.c_str(),
                                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    // NOTE: AutoHandle<hid_t> is object wrapping HDF5 handle so that its automatically
    //       released when object goes out of scope (see RAII pattern).
    //       The class can be found in "base.h".
    {
        // 2. Create new dataset "/Timestep_N/Temperature" which is simulation-domain
        //    sized 2D array of "float"s.
        std::string dataSetName("Temperature");
        // 2.1 Define shape of the dataset (2D edgeSize x edgeSize array).
        AutoHandle<hid_t> dataSpaceHandle(H5Screate_simple(2, gridSize, NULL), H5Sclose);

        hsize_t memSize[]     = {hsize_t(m_tile_size_y), hsize_t(m_tile_size_x)};
        AutoHandle<hid_t> propertyList(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
        H5Pset_chunk(propertyList, 2, memSize);

        // 2.2 Create datased with specified shape.
        AutoHandle<hid_t> dataSetHandle(H5Dcreate(groupHandle, dataSetName.c_str(),
                                                  H5T_NATIVE_FLOAT, dataSpaceHandle,
                                                  H5P_DEFAULT, propertyList, H5P_DEFAULT), H5Dclose);


        AutoHandle<hid_t> memSpaceHandle(H5Screate_simple(2, memSize, NULL), H5Sclose);

        hsize_t slabStart[] = { m_element_displacement[m_rank] / m_materialProperties.GetEdgeSize(),
                                m_element_displacement[m_rank] % m_materialProperties.GetEdgeSize() };

        hsize_t slabSize[]  = {hsize_t(m_tile_size_y), hsize_t(m_tile_size_x)};

        H5Sselect_hyperslab(dataSpaceHandle, H5S_SELECT_SET, slabStart, nullptr, slabSize, nullptr);

        AutoHandle<hid_t> xferPList(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
        H5Pset_dxpl_mpio(xferPList, H5FD_MPIO_COLLECTIVE);

        // Write the data from memory pointed by "data" into new datased.
        H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, xferPList, data);

        // NOTE: Both dataset and dataspace will be closed here automatically (due to RAII).
    }

    {
        // 3. Create Integer attribute in the same group "/Timestep_N/Time"
        //    in which we store number of current simulation iteration.
        std::string attributeName("Time");

        // 3.1 Dataspace is single value/scalar.
        AutoHandle<hid_t> dataSpaceHandle(H5Screate(H5S_SCALAR), H5Sclose);

        // 3.2 Create the attribute in the group as double.
        AutoHandle<hid_t> attributeHandle(H5Acreate2(groupHandle, attributeName.c_str(),
                                                     H5T_IEEE_F64LE, dataSpaceHandle,
                                                     H5P_DEFAULT, H5P_DEFAULT), H5Aclose);

        // 3.3 Write value into the attribute.
        double snapshotTime = double(iteration);
        H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
        
        // NOTE: Both dataspace and attribute handles will be released here.
    }

    // NOTE: The group handle will be released here.
}
