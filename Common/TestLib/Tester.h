
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/TestLib/Tester.h

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

    class Tester
    {

    public:

        enum
        {
            ADAPTER_BASE  = OpenNet::ADAPTER_STATS_QTY,
            ADAPTER_QTY   =   4,
            FUNCTION_QTY  =   2,
            GENERATOR_QTY =   2,
            HARDWARE_BASE = OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY,
            KERNEL_QTY    =   2,
            STATS_QTY     = 128,
        };

        typedef enum
        {
            CODE_FORWARD       ,
            CODE_NONE          ,
            CODE_NOTHING       ,
            CODE_REPLY         ,
            CODE_REPLY_ON_ERROR,
        }
        Code;

        typedef enum
        {
            MODE_FUNCTION,
            MODE_KERNEL  ,

            MODE_QTY
        }
        Mode;

        static void Describe(char aTest);

        static void A_Describe();
        static void B_Describe();
        static void C_Describe();
        static void D_Describe();
        static void E_Describe();
        static void F_Describe();

        Tester(Mode aMode, bool aProfiling);

        ~Tester();

        void SetBandwidth (double       aBandwidth_MiB_s);
        void SetCode      (Code         aCode           );
        void SetMode      (Mode         aMode           );
        void SetPacketSize(unsigned int aPacketSize_byte);

        unsigned int A(unsigned int aBufferQty);
        unsigned int B(unsigned int aBufferQty);
        unsigned int C(unsigned int aBufferQty);
        unsigned int D(unsigned int aBufferQty);
        unsigned int E(unsigned int aBufferQty);
        unsigned int F(unsigned int aBufferQty);

        double       Adapter_GetBandwidth         () const;
        double       Adapter_GetPacketThroughput  () const;

        void DisplaySpeed            ();

        unsigned int Search(char aTest, unsigned int aBufferQty);

        void Start();
        void Stop ();

        unsigned int Test(char aTest, unsigned int aBufferQty);

        unsigned int Verify(char aTest, unsigned int aBufferQty);

    private:

        unsigned int A_Init(unsigned int aBufferQty);
        unsigned int B_Init(unsigned int aBufferQty);
        unsigned int C_Init(unsigned int aBufferQty);
        unsigned int D_Init(unsigned int aBufferQty);
        unsigned int E_Init(unsigned int aBufferQty);
        unsigned int F_Init(unsigned int aBufferQty);

        unsigned int Init0 ();
        unsigned int Init1 ();
        unsigned int Init_2_SameCard();
        unsigned int Init_2_SamePort();
        unsigned int Init_4_SameCard();
        unsigned int Uninit();

        void         Adapter_Connect();
        void         Adapter_InitialiseConstraints();
        void         Adapter_SetProcessing        (OpenNet::Adapter * aAdapter, OpenNet::Function * aFunction, const char * aCode, const char * aName, const char * aIndex);
        void         Adapter_SetProcessing        (OpenNet::Adapter * aAdapter, OpenNet::Kernel   * aKernel  , const char * aCode,                     const char * aIndex);
        unsigned int Adapter_VerifyStatistics     (unsigned int aAdapter);

        void Adapters_RetrieveNo   ();
        void Adapters_SetProcessing();

        void GetAdapterStatistics();

        void Processor_EnableProfiling();

        void ResetAdapterStatistics();

        void ResetInputFilter();

        void SetConfig   ();
        void SetProcessor();

        unsigned int                     mAdapterCount0;
        unsigned int                     mAdapterCount1;
        OpenNet::Adapter               * mAdapters  [ADAPTER_QTY  ];
        Code                             mCodes     [ADAPTER_QTY  ];
        OpenNet::Function                mFunctions [FUNCTION_QTY ];
        OpenNet::Kernel                  mKernels   [KERNEL_QTY   ];
        OpenNet::PacketGenerator::Config mGeneratorConfig;
        unsigned int                     mGeneratorCount;
        OpenNet::PacketGenerator       * mGenerators[GENERATOR_QTY];
        char                             mNos       [ADAPTER_QTY  ][8];

        KmsLib::ValueVector::Constraint_UInt32 mConstraints[STATS_QTY];

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
