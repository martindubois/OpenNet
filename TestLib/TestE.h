
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/TestLib/TestE.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "../Common/TestLib/Test.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class TestE : public TestLib::Test
{

public:

    typedef struct
    {
        unsigned int mBufferEvent;
        unsigned int mEvent      ;
    }
    Counters;

    TestE();

    // ===== TestLib::Test ==================================================
    virtual ~TestE();

    virtual void Info_Display() const;

protected:

    // ===== TestLib::Test ==================================================
    virtual unsigned int Init ();
    virtual unsigned int Start( unsigned int aFlags );
    virtual unsigned int Stop ();

private:

    Counters mCounters;

};
