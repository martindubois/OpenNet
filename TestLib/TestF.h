
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/TestLib/TestF.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "../Common/TestLib/Test.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class TestF : public TestLib::Test
{

public:

    TestF();

    // ===== TestLib::Test ==================================================
    virtual ~TestF();

    virtual void Info_Display() const;

protected:

    // ===== TestLib::Test ==================================================
    virtual unsigned int Init ();
    virtual unsigned int Start();
    virtual unsigned int Stop ();

};
