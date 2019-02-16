
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/TestLib/TestB.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "../Common/TestLib/Test.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class TestB : public TestLib::Test
{

public:

    TestB();

    // ===== TestLib::Test ==================================================
    virtual ~TestB();

    virtual void Info_Display() const;

protected:

    // ===== TestLib::Test ==================================================
    virtual unsigned int Init ();
    virtual unsigned int Start( unsigned int aFlags );
    virtual unsigned int Stop ();

};
