
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

        TestDual(unsigned int aBufferQty0, unsigned int aBufferQty1, bool aProfiling);

        ~TestDual();

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

        void Adapter_Connect();
        void Adapter_Get    ();

        void Processor_EnableProfiling();

        void ResetInputFilter();

        void SetConfig   ();
        void SetProcessor();

        unsigned int         mBufferQty [ADAPTER_QTY];
        OpenNet::Processor * mProcessor         ;
        unsigned int         mStatistics[ADAPTER_QTY][STATS_QTY];
        OpenNet::System    * mSystem            ;

    };

}
