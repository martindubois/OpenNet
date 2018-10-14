
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/TestLib/TestDual.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>
#include <OpenNet/Function_Forward.h>
#include <OpenNet/Kernel_Forward.h>
#include <OpenNet/PacketGenerator.h>
#include <OpenNet/System.h>

// ===== Common =============================================================
#include "../Common/OpenNet/Adapter_Statistics.h"
#include "../Common/OpenNetK/Adapter_Statistics.h"

namespace TestLib
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    class TestDual
    {

    public:

        enum
        {
            ADAPTER_BASE  = OpenNet::ADAPTER_STATS_QTY,
            ADAPTER_QTY   =   2,
            HARDWARE_BASE = OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY,
            STATS_QTY     = 128,
        };

        typedef enum
        {
            ADAPTER_SELECT_CARD_DIFF,
            ADAPTER_SELECT_CARD_SAME,

            ADAPTER_SELECT_QTY
        }
        AdapterSelect;

        typedef enum
        {
            MODE_FUNCTION,
            MODE_KERNEL  ,

            MODE_QTY
        }
        Mode;

        TestDual(Mode aMode, bool aProfiling);

        ~TestDual();

        //     Internel   Ethernet   Internal
        //
        // Dropped <--- 0 <------- 1 <--- Generator
        //
        //                  Send
        //                  0   1   Read    Write   Total
        // Ethernet             1                   1
        // PCIe                     1       1       2
        // Memory - GPU                     1       1
        // Memory - Main            1               1

        unsigned int A       (unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect);
        unsigned int A_Search(unsigned int aBufferQty, unsigned int aPacketSize_byte,                          AdapterSelect aSelect);
        unsigned int A_Verify(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect);

        // Internal   Ethernet   Internal
        //
        //     +---   <-------   <--- Generator
        //     |    0          1
        //     +-->   ------->   ---> Dropped
        //
        //                  Send
        //                  0   1   Read    Write   Total
        // Ethernet         1   1                   2
        // PCIe                     2       2       4
        // Memory - GPU             1       2       3
        // Memory - Main            1               1

        unsigned int B       (unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect);
        unsigned int B_Search(unsigned int aBufferQty, unsigned int aPacketSize_byte,                          AdapterSelect aSelect);
        unsigned int B_Verify(unsigned int aBufferQty, unsigned int aPacketSize_byte, double aBandwidth_MiB_s, AdapterSelect aSelect);

        // TODO  TestLib.TestDual
        //       Mettre privee ce qui peut l'etre

        double       Adapter_GetBandwidth         () const;
        unsigned int Adapter_GetDroppedPacketCount() const;
        double       Adapter_GetPacketThroughput  () const;
        void         Adapter_InitialiseConstraints();
        void         Adapter_SetInputFunctions    ();
        void         Adapter_SetInputFunction     (unsigned int aAdapter);
        void         Adapter_SetInputKernels      ();
        void         Adapter_SetInputKernel       (unsigned int aAdapter);
        unsigned int Adapter_VerifyStatistics     (unsigned int aAdapter);

        void DisplayAdapterStatistics();
        void DisplaySpeed            ();

        void GetAdapterStatistics();

        void GetAndDisplayKernelStatistics();

        void ResetAdapterStatistics();

        void Start();
        void Stop ();

        OpenNet::Adapter               * mAdapters [ADAPTER_QTY];
        OpenNet::Function_Forward        mFunctions[ADAPTER_QTY];
        OpenNet::Kernel_Forward          mKernels  [ADAPTER_QTY];
        OpenNet::PacketGenerator       * mPacketGenerator       ;
        OpenNet::PacketGenerator::Config mPacketGenerator_Config;

        KmsLib::ValueVector::Constraint_UInt32 mConstraints[STATS_QTY];

    private:

        unsigned int A_Init(unsigned int aBufferQty, AdapterSelect aSelect);

        unsigned int B_Init(unsigned int aBufferQty, AdapterSelect aSelect);

        unsigned int Init  (AdapterSelect aSelect);
        unsigned int Uninit();

        void Adapter_Connect();
        void Adapter_Get    (AdapterSelect aSelect);

        void Processor_EnableProfiling();

        void ResetInputFilter();

        void SetConfig   ();
        void SetProcessor();

        double               mBandwidth_MiB_s   ;
        unsigned int         mBufferQty [ADAPTER_QTY];
        Mode                 mMode              ;
        double               mPacketThroughput  ;
        OpenNet::Processor * mProcessor         ;
        bool                 mProfiling         ;
        unsigned int         mStatistics[ADAPTER_QTY][STATS_QTY];
        OpenNet::System    * mSystem            ;

    };

}
