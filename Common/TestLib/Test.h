
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       Common/TestLib/Test.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdint.h>
#include <stdio.h>

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

    class Test
    {

    public:

        typedef enum
        {
            CODE_DEFAULT              ,
            CODE_DO_NOT_REPLY_ON_ERROR,
            CODE_FORWARD              ,
            CODE_NONE                 ,
            CODE_NOTHING              ,
            CODE_REPLY                ,
            CODE_REPLY_ON_ERROR       ,

            CODE_QTY
        }
        Code;

        typedef enum
        {
            MODE_DEFAULT ,
            MODE_FUNCTION,
            MODE_KERNEL  ,

            MODE_QTY
        }
        Mode;

        typedef struct
        {
            double       mBandwidth_MiB_s;
            unsigned int mBufferQty      ;
            Code         mCode           ;
            Mode         mMode           ;
            unsigned int mPacketSize_byte;
            bool         mProfiling      ;
        }
        Config;

        static const double BANDWIDTH_MAX_MiB_s;
        static const double BANDWIDTH_MIN_MiB_s;

        static const unsigned int BUFFER_QTY_MAX;
        static const unsigned int BUFFER_QTY_MIN;

        static const char * CODE_NAMES[CODE_QTY];
        static const char * MODE_NAMES[MODE_QTY];

        static const unsigned char MASK_E[6];
        static const unsigned char MASK_1[6];

        static const unsigned int TEST_PACKET_SIZE_MAX_byte;
        static const unsigned int TEST_PACKET_SIZE_MIN_byte;

        static unsigned int CodeFromName(const char * aName, Code * aOut);
        static unsigned int ModeFromName(const char * aName, Mode * aOut);

        virtual ~Test();

        void SetConfig(const Config & aConfig);

        virtual void Info_Display() const = 0;

        unsigned int Search_Bandwidth ();
        unsigned int Search_BufferQty ();
        unsigned int Search_PacketSize();

        unsigned int Run();

        unsigned int StartStop();

        unsigned int Verify_Bandwidth ();
        unsigned int Verify_BufferQty ();
        unsigned int Verify_PacketSize();

    protected:

        enum
        {
            ADAPTER_BASE  = OpenNet::ADAPTER_STATS_QTY,
            ADAPTER_QTY   =   4,
            HARDWARE_BASE = OpenNet::ADAPTER_STATS_QTY + OpenNetK::ADAPTER_STATS_QTY,
            STATS_QTY     = 128,
        };

        typedef struct
        {
            double mBandwidth_MiB_s          ;
            double mPacketThroughput_packet_s;
        }
        Result;

        static void Connections_Display_1_Card ();
        static void Connections_Display_2_Cards();

        Test(const char * aName, Code aCode, Mode aMode);

        unsigned int               GetAdapterStats(unsigned int aIndex, unsigned int aCounter);
        const Config             * GetConfig    () const;
        OpenNet::PacketGenerator * GetGenerator (unsigned int aIndex);
        OpenNet::System          * GetSystem    ();

        void SetAdapterCount0 (unsigned int aCount);
        void SetAdapterCount1 (unsigned int aCount);
        void SetBufferQty     (unsigned int aIndex, unsigned int aQty );
        void SetCode          (unsigned int aIndex, Code         aCode);
        void SetGeneratorCount(unsigned int aCount);

        void                 DisplayAdapterStats     (unsigned int aIndex);
        virtual unsigned int Init            ();
        void                 InitAdapterConstraints  ();
        virtual unsigned int Start                   (unsigned int aFlags);
        virtual unsigned int Stop            ();
        unsigned int         VerifyAdapterStats  (unsigned int aIndex);

        OpenNet::Adapter                     * mAdapters   [ADAPTER_QTY];
        KmsLib::ValueVector::Constraint_UInt32 mConstraints[STATS_QTY];
        Result                                 mResult;

    private:

        enum
        {
            FUNCTION_QTY  = 2,
            GENERATOR_QTY = 2,
            KERNEL_QTY    = 2,
        };

        typedef struct
        {
            Code mCode;
            Mode mMode;
        }
        Default;

        void ConfigAdapters  ();
        void ConfigGenerators();
        void ConfigProcessor ();

        void DisplayResult();

        void DisplayAndWriteResult(const char * aNote);

        unsigned int Execute( unsigned int aFlags );

        unsigned int InitAndPrepare();

        void Prepare();

        void ResetInputFilters();
        void ResetStatistics  ();

        void RetrieveStatistics();

        void SetFunction(unsigned int aAdapterIndex);
        void SetFunction(OpenNet::Adapter * aAdapter, OpenNet::Function * aFunction, const char * aCode, const char * aName, const char * aIndex);
        void SetKernel  (unsigned int aAdapterIndex);
        void SetKernel  (OpenNet::Adapter * aAdapter, OpenNet::Kernel   * aKernel  , const char * aCode,                     const char * aIndex);

        void Uninit();

        void WriteResult(             const char * aNote) const;
        void WriteResult(FILE * aOut, const char * aNote) const;

        unsigned int               mAdapterCount0 ;
        unsigned int               mAdapterCount1 ;
        unsigned int               mAdapterStats[ADAPTER_QTY][STATS_QTY];
        unsigned int               mBufferQty [ADAPTER_QTY];
        Code                       mCodes     [ADAPTER_QTY];
        Config                     mConfig        ;
        Default                    mDefault       ;
        OpenNet::Function          mFunctions [FUNCTION_QTY];
        unsigned int               mGeneratorCount;
        OpenNet::PacketGenerator * mGenerators[GENERATOR_QTY];
        OpenNet::Kernel            mKernels   [KERNEL_QTY];
        char                       mName      [16];
        char                       mNos       [ADAPTER_QTY][8];
        unsigned int               mPriorityClass ;
        OpenNet::Processor       * mProcessor;
        OpenNet::System          * mSystem;
        
    };

}
