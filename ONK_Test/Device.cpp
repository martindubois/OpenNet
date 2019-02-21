
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Test/Device.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <stdint.h>

#ifdef _KMS_LINUX_
    #include <fcntl.h>
#endif

#ifdef _KMS_WINDOWS_
    // ===== Windows ============================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/DriverHandle.h>
#include <KmsLib/Exception.h>
#include <KmsLib/ThreadBase.h>
#include <KmsTest.h>

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_Types.h>
#include <OpenNetK/PacketGenerator_Types.h>

#ifdef _KMS_WINDOWS_
    #include <OpenNetK/Interface.h>
#endif

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/IoCtl.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void Display( const OpenNetK::Adapter_Config & aConfig );

static unsigned int TestError( KmsLib::DriverHandle * aDH, unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte, unsigned int aExpected );

static unsigned int ValidateStats0( const unsigned int * aStats, unsigned int aSize_byte );
static unsigned int ValidateStats1( const unsigned int * aStats, unsigned int aSize_byte );

// Tests
/////////////////////////////////////////////////////////////////////////////

// TEST INFO  Device
//            Invalid IoCtl code<br>
//            No expected input buffer<br>
//            Invalid event handle
KMS_TEST_BEGIN(Device_SetupA)
{
    uint8_t                          lBuffer[1024];
    OpenNetK::Adapter_Config         lConfig ;
    IoCtl_Connect_In                 lConnect;
    OpenNetK::PacketGenerator_Config lPacketGenConfig;
    IoCtl_Packet_Send_Ex_In        * lPacketSendExIn = reinterpret_cast< IoCtl_Packet_Send_Ex_In * >( lBuffer );
    KmsLib::DriverHandle             lDH0    ;

    memset( & lBuffer      , 0xff, sizeof( lBuffer                 ) );
    memset( lPacketSendExIn,    0, sizeof( IoCtl_Packet_Send_Ex_In ) );

    #ifdef _KMS_LINUX_
        lDH0.Connect("/dev/OpenNet0", O_RDWR);
    #endif

    #ifdef _KMS_WINDOWS_
        lDH0.Connect(OPEN_NET_DRIVER_INTERFACE, 0, GENERIC_ALL, 0);
    #endif

    KMS_TEST_COMPARE( 0, TestError( & lDH0, 0, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    // ====== IOCTL_CONFIG_GET ==============================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_CONFIG_GET, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    lDH0.Control( IOCTL_CONFIG_GET, NULL, 0, & lConfig, sizeof( lConfig ) );

    Display( lConfig );

    KMS_TEST_ASSERT( PACKET_SIZE_MAX_byte >= lConfig.mPacketSize_byte );
    KMS_TEST_ASSERT( PACKET_SIZE_MIN_byte <= lConfig.mPacketSize_byte );

    // ====== IOCTL_CONFIG_SET ==============================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_CONFIG_SET, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    #ifdef _KMS_WINDOWS_
        KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_CONFIG_SET, & lConfig, sizeof( lConfig ), NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );
    #endif

    // ===== IOCTL_CONNECT ==================================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_CONNECT, NULL      ,                  0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_CONNECT, & lConnect, sizeof( lConnect ), NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    // ===== IOCTL_INFO_GET =================================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_INFO_GET, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    // ====== IOCTL_PACKET_SEND =============================================

    // ====== IOCTL_PACKET_SEND_EX ==========================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_PACKET_SEND_EX, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    lPacketSendExIn->mRepeatCount = 1;
    lPacketSendExIn->mSize_byte   = sizeof( lBuffer ) - sizeof(IoCtl_Packet_Send_Ex_In );

    lDH0.Control( IOCTL_PACKET_SEND_EX, lPacketSendExIn, sizeof( lBuffer ), NULL, 0 );

    // ===== IOCTL_START ====================================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_START, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    // ===== IOCTL_STATE_GET ================================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_STATE_GET, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    // ====== IOCTL_STATISTICS_GET ==========================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_STATISTICS_GET, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    // ===== IOCTL_STATISTICS_RESET =========================================

    // ===== IOCTL_STOP =====================================================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_STOP, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    // ===== IOCTL_PACKET_DROP ==============================================
    lDH0.Control( IOCTL_PACKET_DROP, NULL, 0, NULL, 0 );

    // ===== IOCTL_PACKET_GENERATOR_CONFIG_GET ==============================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_PACKET_GENERATOR_CONFIG_GET, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    lDH0.Control( IOCTL_PACKET_GENERATOR_CONFIG_GET, NULL, 0, & lPacketGenConfig, sizeof( lPacketGenConfig ) );

    // ===== IOCTL_PACKET_GENERATOR_CONFIG_SET ==============================
    KMS_TEST_COMPARE( 0, TestError( & lDH0, IOCTL_PACKET_GENERATOR_CONFIG_SET, NULL, 0, NULL, 0, KmsLib::Exception::CODE_IOCTL_ERROR ) );

    lPacketGenConfig.mAllowedIndexRepeat =  1;
    lPacketGenConfig.mIndexOffset_byte   =  0;
    lPacketGenConfig.mPacketPer100ms     =  1;
    lPacketGenConfig.mPacketSize_byte    = 96;

    lDH0.Control( IOCTL_PACKET_GENERATOR_CONFIG_SET, & lPacketGenConfig, sizeof( lPacketGenConfig ), & lPacketGenConfig, sizeof( lPacketGenConfig ) );

    // ===== IOCTL_PACKET_GENERATOR_START ===================================
    lDH0.Control( IOCTL_PACKET_GENERATOR_START, NULL, 0, NULL, 0 );

    // ===== IOCTL_PACKET_GENERATOR_STOP ====================================

    KmsLib::ThreadBase::Sleep_ms( 200 );

    lDH0.Control( IOCTL_PACKET_GENERATOR_STOP , NULL, 0, NULL, 0 );
}
KMS_TEST_END_2

// TEST INFO  Device
//            Send and Receive
KMS_TEST_BEGIN( Device_SetupB )
{
    uint8_t                   lBuffer[1024];
    KmsLib::DriverHandle      lDH0;
    KmsLib::DriverHandle      lDH1;
    IoCtl_Packet_Send_Ex_In * lPacketSendExIn = reinterpret_cast< IoCtl_Packet_Send_Ex_In * >( lBuffer );
    IoCtl_Statistics_Get_In   lStatsGetIn;
    unsigned int              lStats[ 128 ];

    memset( & lBuffer      , 0xff, sizeof( lBuffer                 ) );
    memset( lPacketSendExIn,    0, sizeof( IoCtl_Packet_Send_Ex_In ) );
    memset( & lStats       ,    0, sizeof( lStats                  ) );
    memset( & lStatsGetIn  ,    0, sizeof( lStatsGetIn             ) );

    #ifdef _KMS_LINUX_
        lDH0.Connect("/dev/OpenNet0", O_RDWR);
        lDH1.Connect("/dev/OpenNet1", O_RDWR);
    #endif

    #ifdef _KMS_WINDOWS_
        lDH0.Connect(OPEN_NET_DRIVER_INTERFACE, 0, GENERIC_ALL, 0);
        lDH1.Connect(OPEN_NET_DRIVER_INTERFACE, 1, GENERIC_ALL, 0);
    #endif

    lDH0.Control( IOCTL_STATISTICS_RESET, NULL, 0, NULL, 0 );
    lDH1.Control( IOCTL_STATISTICS_RESET, NULL, 0, NULL, 0 );

    lDH0.Control( IOCTL_PACKET_DROP, NULL, 0, NULL, 0 );

    lPacketSendExIn->mRepeatCount = 1;
    lPacketSendExIn->mSize_byte   = sizeof( lBuffer ) - sizeof(IoCtl_Packet_Send_Ex_In );

    lDH1.Control( IOCTL_PACKET_SEND_EX, lPacketSendExIn, sizeof( lBuffer ), NULL, 0 );

    KmsLib::ThreadBase::Sleep_ms( 200 );

    lStatsGetIn.mFlags.mReset    = false;
    lStatsGetIn.mOutputSize_byte = sizeof( lStats );

    lDH0.Control( IOCTL_STATISTICS_GET, & lStatsGetIn, sizeof( lStatsGetIn ), lStats, sizeof( lStats ) );
    KMS_TEST_COMPARE( 0, ValidateStats0( lStats, sizeof( lStats ) ) );

    lDH1.Control( IOCTL_STATISTICS_GET, & lStatsGetIn, sizeof( lStatsGetIn ), lStats, sizeof( lStats ) );
    KMS_TEST_COMPARE( 0, ValidateStats1( lStats, sizeof( lStats ) ) );
}
KMS_TEST_END_2

// Static function
/////////////////////////////////////////////////////////////////////////////

void Display( const OpenNetK::Adapter_Config & aConfig )
{
    assert( NULL != ( & aConfig ) );

    printf( "OpenNetK::Adapter_Config :\n");
    printf( "    PacketSize : %u bytes\n", aConfig.mPacketSize_byte );
}

unsigned int TestError( KmsLib::DriverHandle * aDH, unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte, unsigned int aExpected )
{
    assert( NULL != aDH );

    try
    {
        aDH->Control( aCode, aIn, aInSize_byte, aOut, aOutSize_byte );

        return __LINE__;
    }
    catch ( KmsLib::Exception * eE )
    {
        assert( NULL != eE );

        KMS_TEST_ERROR_INFO;
        eE->Write( stdout );

        if ( aExpected != eE->GetCode() )
        {
            printf( "Expected code = %u, Indicated code = %u\n", aExpected, eE->GetCode() );
            return __LINE__;
        }
    }

    return 0;
}

unsigned int ValidateStats0( const unsigned int * aStats, unsigned int aSize_byte )
{
    assert( NULL != aStats     );
    assert(    0 <  aSize_byte );

    unsigned int lCount  = aSize_byte / sizeof( unsigned int );
    unsigned int lResult = 0;

    for ( unsigned int i = 0; i < lCount; i ++ )
    {
        switch ( i )
        {
        case 0 : // ADAPTER_STATS_BUFFERS_PROCESS
            if ( 3 != aStats[ i ] )
            {
                printf( "ERROR  Counter %3u = %3u, Expected   3\n", i, aStats[ i ] );
                lResult ++;
            }
            break;

        case 16 : // ADAPTER_STATS_RUNNING_TIME_ms
        case 29 : // ADAPTER_STATS_PACKET_GENERATOR_REPEAT_COUNT
        case 31 : // ADAPTER_STATS_IOCTL_STATISTICS_RESET
        case 93 : // HARDWARE_STATS_INTERRUPT_PROCESS_LAST_MESSAGE_ID
            break;

        case 36 : // HARDWARE_STATS_INTERRUPT_PROCESS
        case 38 : // HARDWARE_STATS_PACKET_RECEIVE
        case 40 : // HARDWARE_STATS_RX_packet
        case 62 : // HARDWARE_STATS_RX_HOST_packet
            if ( 1 != aStats[ i ] )
            {
                printf( "ERROR  Counter %3u = %3u, Expected   1\n", i, aStats[ i ] );
                lResult ++;
            }
            break;

        case 61 : // HARDWARE_STATS_RX_HOST_byte
            if ( 996 != aStats[ i ] )
            {
                printf( "ERROR  Counter %3u = %3u, Expected 996\n", i, aStats[ i ] );
                lResult ++;
            }
            break;

        default :
            if ( 0 != aStats[ i ] )
            {
                printf( "ERROR  Counter %3u = %3u, Expected   0\n", i, aStats[ i ] );
                lResult ++;
            }
        }
    }

    return lResult;
}

unsigned int ValidateStats1( const unsigned int * aStats, unsigned int aSize_byte )
{
    assert( NULL != aStats     );
    assert(    0 <  aSize_byte );

    unsigned int lCount  = aSize_byte / sizeof( unsigned int );
    unsigned int lResult = 0;

    for ( unsigned int i = 0; i < lCount; i ++ )
    {
        switch ( i )
        {
        case 0 : // ADAPTER_STATS_BUFFERS_PROCESS
            if ( 3 != aStats[ i ] )
            {
                printf( "ERROR  Counter %3u = %3u, Expected   3\n", i, aStats[ i ] );
                lResult ++;
            }
            break;

        case 11 : // ADAPTER_STATS_IOCTL_PACKET_SEND
        case 36 : // HARDWARE_STATS_INTERRUPT_PROCESS
        case 39 : // HARDWARE_STATS_PACKET_SEND
        case 43 : // HARDWARE_STATS_TX_packet
        case 75 : // HARDWARE_STATS_TX_HOST_packet
            if ( 1 != aStats[ i ] )
            {
                printf( "ERROR  Counter %3u = %3u, Expected   1\n", i, aStats[ i ] );
                lResult ++;
            }
            break;

        case 16 : // ADAPTER_STATS_RUNNING_TIME_ms
        case 31 : // ADAPTER_STATS_IOCTL_STATISTICS_RESET
        case 93 : // HARDWARE_STATS_INTERRUPT_PROCESS_LAST_MESSAGE_ID
            break;

        case 74 : // HARDWARE_STATS_TX_HOST_byte
            if ( 996 != aStats[ i ] )
            {
                printf( "ERROR  Counter %3u = %3u, Expected 996\n", i, aStats[ i ] );
                lResult ++;
            }
            break;

        default :
            if ( 0 != aStats[ i ] )
            {
                printf( "ERROR  Counter %3u = %3u, Expected   0\n", i, aStats[ i ] );
                lResult ++;
            }
        }
    }

    return lResult;
}
