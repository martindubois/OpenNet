
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestFactory.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <string.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

#include "../Common/TestLib/TestFactory.h"

// ===== TestLib ============================================================
#include "Code.h"
#include "TestA.h"
#include "TestB.h"
#include "TestC.h"
#include "TestD.h"
#include "TestE.h"
#include "TestF.h"

namespace TestLib
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    TestFactory::TestFactory()
    {
        ResetConfig();
    }

    TestFactory::~TestFactory()
    {
    }

    // aBandwidth_MiB_s
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetBandwidth(double aBandwidth_MiB_s)
    {
        assert(Test::BANDWIDTH_MAX_MiB_s >= mConfig.mBandwidth_MiB_s);
        assert(Test::BANDWIDTH_MIN_MiB_s <= mConfig.mBandwidth_MiB_s);

        if (Test::BANDWIDTH_MIN_MiB_s > aBandwidth_MiB_s)
        {
            printf(__FUNCTION__ " - %f MiB/s is too slow\n", aBandwidth_MiB_s);
            return __LINE__;
        }

        if (Test::BANDWIDTH_MAX_MiB_s < aBandwidth_MiB_s)
        {
            printf(__FUNCTION__ "- %f MiB/s is too fast\n", aBandwidth_MiB_s);
            return __LINE__;
        }

        mConfig.mBandwidth_MiB_s = aBandwidth_MiB_s;

        return 0;
    }

    // aBandwidth_MiB_s [---;R--]
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetBandwidth(const char * aBandwidth_MiB_s)
    {
        assert(NULL != aBandwidth_MiB_s);

        unsigned int lBandwidth_MiB_s;

        if (1 != sscanf_s(aBandwidth_MiB_s, "%u", &lBandwidth_MiB_s))
        {
            printf(__FUNCTION__ " - %s is not a valid bandwidth\n", aBandwidth_MiB_s);
            return __LINE__;
        }

        return SetBandwidth(lBandwidth_MiB_s);
    }

    // aBufferQty
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetBufferQty(unsigned int aBufferQty)
    {
        assert(Test::BUFFER_QTY_MAX >= mConfig.mBufferQty);
        assert(Test::BUFFER_QTY_MIN <= mConfig.mBufferQty);

        if (Test::BUFFER_QTY_MIN > aBufferQty)
        {
            printf(__FUNCTION__ " - %u buffer is not enough\n", aBufferQty);
            return __LINE__;
        }

        if (Test::BUFFER_QTY_MAX < aBufferQty)
        {
            printf(__FUNCTION__ " - %u buffer is too many\n", aBufferQty);
            return __LINE__;
        }

        mConfig.mBufferQty = aBufferQty;

        return 0;
    }

    // aBufferQty [---;R--]
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetBufferQty(const char * aBufferQty)
    {
        assert(NULL != aBufferQty);

        unsigned int lBufferQty;

        if (1 != sscanf_s(aBufferQty, "%u", &lBufferQty))
        {
            printf(__FUNCTION__ " - %s is not a valid buffer quantity\n", aBufferQty);
            return __LINE__;
        }

        return SetBufferQty(lBufferQty);
    }

    // aCode
    //
    // Return
    //     0  OK
    // Ohter  Error
    void TestFactory::SetCode(Code aCode)
    {
        assert(CODE_QTY > aCode);

        assert(CODE_QTY > mConfig.mCode);

        mConfig.mCode = aCode;
    }

    // aCode [---;R--]
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetCode(const char * aCode)
    {
        assert(NULL != aCode);

        assert(CODE_QTY > mConfig.mCode);

        unsigned int lResult = Test::CodeFromName(aCode, &mConfig.mCode);

        assert(CODE_QTY > mConfig.mCode);

        return lResult;
    }

    // aMode
    //
    // Return
    //     0  OK
    // Ohter  Error
    void TestFactory::SetMode(Test::Mode aMode)
    {
        assert(Test::MODE_QTY > aMode);

        assert(Test::MODE_QTY > mConfig.mMode);

        mConfig.mMode = aMode;
    }

    // aMode [---;R--]
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetMode(const char * aMode)
    {
        assert(NULL != aMode);

        assert(Test::MODE_QTY > mConfig.mMode);

        unsigned int lResult = Test::ModeFromName(aMode, &mConfig.mMode);

        assert(Test::MODE_QTY > mConfig.mMode);

        return lResult;
    }

    // aPacketSize_byte
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetPacketSize(unsigned int aPacketSize_byte)
    {
        assert(Test::TEST_PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);
        assert(Test::TEST_PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte);

        if (Test::TEST_PACKET_SIZE_MIN_byte > aPacketSize_byte)
        {
            printf(__FUNCTION__ " - %u bytes is too small\n", aPacketSize_byte);
            return __LINE__;
        }

        if (Test::TEST_PACKET_SIZE_MAX_byte < aPacketSize_byte)
        {
            printf(__FUNCTION__ " - %u bytes is too large\n", aPacketSize_byte);
            return __LINE__;
        }

        mConfig.mPacketSize_byte = aPacketSize_byte;

        return 0;
    }

    // aPacketSize_byte [---;R--]
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetPacketSize(const char * aPacketSize_byte)
    {
        assert(NULL != aPacketSize_byte);

        unsigned int lPacketSize_byte;

        if (1 != sscanf_s(aPacketSize_byte, "%u", &lPacketSize_byte))
        {
            printf(__FUNCTION__ " - %s is not a valid packet size\n", aPacketSize_byte);
            return __LINE__;
        }

        return SetPacketSize(lPacketSize_byte);
    }

    // aProfiling
    void TestFactory::SetProfiling(bool aProfiling)
    {
        mConfig.mProfiling = aProfiling;
    }

    // aProfiling [---;R--]
    //
    // Return
    //     0  OK
    // Ohter  Error
    unsigned int TestFactory::SetProfiling(const char * aProfiling)
    {
        if (0 == _strnicmp("false", aProfiling, 5))
        {
            mConfig.mProfiling = false;
        }
        else if (0 == _strnicmp("true", aProfiling, 4))
        {
            mConfig.mProfiling = true;
        }
        else
        {
            printf(__FUNCTION__ " - %s is not a valid value\n", aProfiling);
            return __LINE__;
        }

        return 0;
    }

    // aName [---;R--]
    //
    // Return
    //  NULL   Invalid name
    //  Ohter  A new Test instance
    //
    // TestFactory::CreateTest ==> delete
    Test * TestFactory::CreateTest(const char * aName)
    {
        assert(NULL != aName);

        Test * lResult = NULL;

        // new ==> delete
        if (0 == _strnicmp("A", aName, 1)) { lResult = new TestA(); }
        if (0 == _strnicmp("B", aName, 1)) { lResult = new TestB(); }
        if (0 == _strnicmp("C", aName, 1)) { lResult = new TestC(); }
        if (0 == _strnicmp("D", aName, 1)) { lResult = new TestD(); }
        if (0 == _strnicmp("E", aName, 1)) { lResult = new TestE(); }
        if (0 == _strnicmp("F", aName, 1)) { lResult = new TestF(); }

        if (NULL != lResult)
        {
            lResult->SetConfig(mConfig);
        }

        return lResult;
    }

    void TestFactory::DisplayConfig()
    {
        assert(Test::BANDWIDTH_MAX_MiB_s       >= mConfig.mBandwidth_MiB_s);
        assert(Test::BANDWIDTH_MIN_MiB_s       <= mConfig.mBandwidth_MiB_s);
        assert(Test::BUFFER_QTY_MAX            >= mConfig.mBufferQty      );
        assert(Test::BUFFER_QTY_MIN            <= mConfig.mBufferQty      );
        assert(CODE_QTY                        >  mConfig.mCode           );
        assert(Test::MODE_QTY                  >  mConfig.mMode           );
        assert(Test::TEST_PACKET_SIZE_MAX_byte >= mConfig.mPacketSize_byte);
        assert(Test::TEST_PACKET_SIZE_MIN_byte <= mConfig.mPacketSize_byte);

        printf(
            "Bandwidth  = %f MiB/s\n"
            "BufferQty  = %u\n"
            "Code       = %s\n"
            "Mode       = %s\n"
            "PacketSize = %u bytes\n"
            "Profiling  = %s\n",
            mConfig.mBandwidth_MiB_s,
            mConfig.mBufferQty      ,
            CODES[mConfig.mCode].mName,
            Test::MODE_NAMES[mConfig.mMode],
            mConfig.mPacketSize_byte,
            mConfig.mProfiling ? "true" : "false");
    }

    void TestFactory::ResetConfig()
    {
        mConfig.mBandwidth_MiB_s = Test::BANDWIDTH_MAX_MiB_s;
        mConfig.mBufferQty       =                         2;
        mConfig.mCode            = CODE_DEFAULT             ;
        mConfig.mMode            = Test::MODE_DEFAULT       ;
        mConfig.mPacketSize_byte =                      1024;
        mConfig.mProfiling       = false                    ;
    }

}
