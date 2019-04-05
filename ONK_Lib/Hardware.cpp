
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Lib/Hardware.cpp

#define __CLASS__     "Hardware::"
#define __NAMESPACE__ "OpenNetK::"

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/OS.h>
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Adapter.h>
#include <OpenNetK/Hardware_Statistics.h>
#include <OpenNetK/SpinLock.h>

#include <OpenNetK/Hardware.h>

// ===== Common =============================================================
#include "../Common/Constants.h"
#include "../Common/Version.h"

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void * Hardware::operator new(size_t aSize_byte, void * aAddress)
    {
        ASSERT(sizeof(Hardware) <= aSize_byte);
        ASSERT(NULL             != aAddress  );

        (void)(aSize_byte);

        return aAddress;
    }

    unsigned int Hardware::GetPacketSize() const
    {
        return mConfig.mPacketSize_byte;
    }

    void Hardware::ResetMemory()
    {
    }

    void Hardware::SetAdapter(Adapter * aAdapter)
    {
        ASSERT(NULL != aAdapter);

        ASSERT(NULL == mAdapter);

        mAdapter = aAdapter;
    }

    // NOT TESTED  ONK_Lib.Hardware
    //             The SetCommonBuffer is only there to fill the virtual
    //             table entry when the driver does not need a common buffer.
    void Hardware::SetCommonBuffer(uint64_t aCommon_PA, void * aCommon_CA)
    {
        ASSERT(NULL != aCommon_CA);

        (void)(aCommon_PA);
        (void)(aCommon_CA);

        ASSERT(false);
    }

    void Hardware::SetConfig(const Adapter_Config & aConfig)
    {
        ASSERT(NULL != (&aConfig));

        memcpy(&mConfig, &aConfig, sizeof(mConfig));

        mStatistics[HARDWARE_STATS_SET_CONFIG] ++;
    }

    bool Hardware::SetMemory(unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte)
    {
        ASSERT(NULL != aMemory_MA);
        ASSERT(   0 <  aSize_byte);

        (void)(aIndex    );
        (void)(aMemory_MA);
        (void)(aSize_byte);

        return true;
    }

    void Hardware::D0_Entry()
    {
        mStatistics[HARDWARE_STATS_D0_ENTRY] ++;
    }

    bool Hardware::D0_Exit()
    {
        mStatistics[HARDWARE_STATS_D0_EXIT] ++;

        return true;
    }

    void Hardware::Interrupt_Disable()
    {
        mStatistics[HARDWARE_STATS_INTERRUPT_DISABLE] ++;
    }

    void Hardware::Interrupt_Enable()
    {
        mStatistics[HARDWARE_STATS_INTERRUPT_ENABLE] ++;
    }

    // NOT TESTED  ONK_Lib.Hardware
    //             The Interrupt_Process is only there to fill the virtual
    //             table entry when the driver does not need interrupt.
    bool Hardware::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
    {
        ASSERT(NULL != aNeedMoreProcessing);

        (void)(aMessageId         );
        (void)(aNeedMoreProcessing);

        ASSERT(false);

        return false;
    }

    // CRITICAL PATH  Interrupt
    //                1 / hardware interrupt + 1 / tick
    void Hardware::Interrupt_Process2(bool * aNeedMoreProcessing)
    {
        // printk( KERN_DEBUG "%s(  )\n", __FUNCTION__ );

        ASSERT(NULL != aNeedMoreProcessing);

        ASSERT(NULL != mAdapter);

        mAdapter->Interrupt_Process2(aNeedMoreProcessing);
    }

    void Hardware::Interrupt_Process3()
    {
        ASSERT(NULL != mAdapter);

        mAdapter->Interrupt_Process3();
    }

    void Hardware::Unlock_AfterReceive_FromThread(volatile long * aCounter, unsigned int aPacketQty, uint32_t aFlags )
    {
        // TRACE_DEBUG "Unlock_AfterReceive( , %u packet )" DEBUG_EOL, aPacketQty TRACE_END;

        ASSERT(NULL != aCounter );
        ASSERT(0    < aPacketQty);

        ( * aCounter ) += aPacketQty;

        Unlock_AfterReceive_Internal();

        mStatistics[OpenNetK::HARDWARE_STATS_PACKET_RECEIVE] += aPacketQty;

        mZone0->UnlockFromThread( aFlags );
    }

    void Hardware::Unlock_AfterSend_FromThread(volatile long * aCounter, unsigned int aPacketQty, uint32_t aFlags )
    {
        ASSERT(NULL != aCounter  );
        ASSERT(   0 <  aPacketQty);

        ( * aCounter ) += aPacketQty;

        Unlock_AfterSend_Internal();

        mStatistics[OpenNetK::HARDWARE_STATS_PACKET_SEND] += aPacketQty;

        mZone0->UnlockFromThread( aFlags );
    }

    unsigned int Hardware::Statistics_Get(uint32_t * aOut, unsigned int aOutSize_byte, bool aReset)
    {
        unsigned int lResult_byte;

        if (sizeof(mStatistics) <= aOutSize_byte)
        {
            memcpy(aOut, &mStatistics, aOutSize_byte);
            lResult_byte = sizeof(mStatistics);
        }
        else
        {
            if (0 < aOutSize_byte)
            {
                memcpy(aOut, &mStatistics, aOutSize_byte);
                lResult_byte = aOutSize_byte;
            }
            else
            {
                lResult_byte = 0;
            }
        }

        if (aReset)
        {
            memset(&mStatistics, 0, sizeof(uint32_t) * HARDWARE_STATS_RESET_QTY);
        }

        mStatistics[HARDWARE_STATS_STATISTICS_GET] ++;

        return lResult_byte;
    }

    void Hardware::Statistics_Reset()
    {
        memset(&mStatistics, 0, sizeof(uint32_t) * HARDWARE_STATS_RESET_QTY);
    }

    void Hardware::Tx_Disable()
    {
        mTx_Enabled = false;
    }

    void Hardware::Tx_Enable()
    {
        mTx_Enabled = true;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    void Hardware::Init(SpinLock * aZone0)
    {
        ASSERT(NULL != aZone0);

        ASSERT(NULL == mZone0);

        mZone0 = aZone0;
    }

    unsigned int Hardware::GetCommonBufferSize() const
    {
        return mInfo.mCommonBufferSize_byte;
    }

    void Hardware::GetConfig(Adapter_Config * aConfig)
    {
        ASSERT(NULL != aConfig);

        memcpy(aConfig, &mConfig, sizeof(mConfig));
    }

    void Hardware::GetInfo(Adapter_Info * aInfo)
    {
        ASSERT(NULL != aInfo);

        memcpy(aInfo, &mInfo, sizeof(mInfo));
    }

    // aCounter [---;RW-]
    // aPacketQty         The number of packet programmed for receiving
    //
    // Level  SoftInt

    // CRITICAL PATH  Interrupt.Rx
    //                1 / buffer
    void Hardware::Unlock_AfterReceive(volatile long * aCounter, unsigned int aPacketQty)
    {
        // TRACE_DEBUG "Unlock_AfterReceive( , %u packet )" DEBUG_EOL, aPacketQty TRACE_END;

        ASSERT(NULL != aCounter );
        ASSERT(0    < aPacketQty);

        ASSERT( NULL != mZone0 );

        ( * aCounter ) += aPacketQty;

        Unlock_AfterReceive_Internal();

        mStatistics[OpenNetK::HARDWARE_STATS_PACKET_RECEIVE] += aPacketQty;

        mZone0->Unlock();
    }

    // aCounter [---;RW-]
    // aPacketQty         The number of packet programmed for sending
    //
    // Level  SoftInt

    // CRITICAL PATH  Interrupt.Tx
    //                1 / buffer
    void Hardware::Unlock_AfterSend(volatile long * aCounter, unsigned int aPacketQty)
    {
        // TRACE_DEBUG "Unlock_AfterSend( , %u packet )" DEBUG_EOL, aPacketQty TRACE_END;

        ASSERT(NULL != aCounter  );
        ASSERT(   0 <  aPacketQty);

        ASSERT( NULL != mZone0 );

        ( * aCounter ) += aPacketQty;

        Unlock_AfterSend_Internal();

        mStatistics[OpenNetK::HARDWARE_STATS_PACKET_SEND] += aPacketQty;

        mZone0->Unlock();
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    void Hardware::SkipDangerousBoundary(uint64_t * aIn_PA, uint8_t ** aIn_XA, unsigned int aSize_byte, uint64_t * aOut_PA, uint8_t ** aOut_XA)
    {
        ASSERT(NULL                                  !=   aIn_PA     );
        ASSERT(NULL                                  !=   aIn_XA     );
        ASSERT(NULL                                  != (*aIn_XA    ));
        ASSERT(                                    0 <    aSize_byte );
        ASSERT(OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte >    aSize_byte );
        ASSERT(NULL                                  !=   aOut_PA    );
        ASSERT(NULL                                  !=   aOut_XA    );

        uint64_t lEnd = (*aIn_PA) + aSize_byte - 1;

        if (((*aIn_PA) & OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte) == (lEnd & OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte))
        {
            (*aOut_PA) = (*aIn_PA);
            (*aOut_XA) = (*aIn_XA);
        }
        else
        {
            uint64_t lOffset_byte = OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte - ((*aIn_PA) % OPEN_NET_DANGEROUS_BOUNDARY_SIZE_byte);

            (*aOut_PA) = (*aIn_PA) + lOffset_byte;
            (*aOut_XA) = (*aIn_XA) + lOffset_byte;
        }

        (*aIn_PA) = (*aOut_PA) + aSize_byte;
        (*aIn_XA) = (*aOut_XA) + aSize_byte;
    }

    Hardware::Hardware(OpenNetK::Adapter_Type aType, unsigned int aPacketSize_byte) : mAdapter(NULL), mTx_Enabled(true), mZone0(NULL)
    {
        ASSERT(OpenNetK::ADAPTER_TYPE_UNKNOWN != aType           );
        ASSERT(PACKET_SIZE_MAX_byte           >= aPacketSize_byte);
        ASSERT(PACKET_SIZE_MIN_byte           <= aPacketSize_byte);

        memset(&mConfig, 0, sizeof(mConfig));
        memset(&mInfo  , 0, sizeof(mInfo  ));

        mConfig.mPacketSize_byte = aPacketSize_byte;

        mInfo.mAdapterType     = aType           ;
        mInfo.mPacketSize_byte = aPacketSize_byte;

        mInfo.mVersion_Driver.mMajor         = VERSION_MAJOR;
        mInfo.mVersion_Driver.mMinor         = VERSION_MINOR;
        mInfo.mVersion_Driver.mBuild         = VERSION_BUILD;
        mInfo.mVersion_Driver.mCompatibility = VERSION_COMPATIBILITY;
    
        mInfo.mVersion_ONK_Lib.mMajor         = VERSION_MAJOR;
        mInfo.mVersion_ONK_Lib.mMinor         = VERSION_MINOR;
        mInfo.mVersion_ONK_Lib.mBuild         = VERSION_BUILD;
        mInfo.mVersion_ONK_Lib.mCompatibility = VERSION_COMPATIBILITY;
    
        strcpy(mInfo.mVersion_Driver .mType       , VERSION_TYPE  );
        strcpy(mInfo.mVersion_ONK_Lib.mCompiled_At, __TIME__      );
        strcpy(mInfo.mVersion_ONK_Lib.mCompiled_On, __DATE__      );
        strcpy(mInfo.mVersion_ONK_Lib.mComment    , "ONK_Lib"     );
        strcpy(mInfo.mVersion_ONK_Lib.mType       , VERSION_TYPE  );
    }

}
