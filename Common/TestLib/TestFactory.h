
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       Common/TestLib/TestFactory.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "Code.h"
#include "Test.h"

namespace TestLib
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    class TestFactory
    {

    public:

        TestFactory();

        ~TestFactory();

        unsigned int SetBandwidth (double       aBandwidth_MiB_s);
        unsigned int SetBandwidth (const char * aBandwidth_MiB_s);
        unsigned int SetBufferQty (unsigned int aBufferQty      );
        unsigned int SetBufferQty (const char * aBufferQty      );
        void         SetCode      (Code         aCode           );
        unsigned int SetCode      (const char * aCode           );
        void         SetMode      (Test::Mode   aMode           );
        unsigned int SetMode      (const char * aMode           );
        unsigned int SetPacketSize(unsigned int aPacketSize_byte);
        unsigned int SetPacketSize(const char * aPacketSize_byte);
        void         SetProfiling (bool         aProfiling      );
        unsigned int SetProfiling (const char * aProfiling      );

        Test * CreateTest(const char * aName);

        void DisplayConfig();
        void ResetConfig  ();

    private:

        Test::Config mConfig;

    };

}
