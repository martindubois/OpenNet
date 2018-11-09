
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     Common/TestLib/TestD.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "../Common/TestLib/Test.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class TestD : public TestLib::Test
{

public:

    TestD();

    // ===== TestLib::Test ==================================================
    virtual ~TestD();

    virtual void Info_Display() const;

protected:

    // ===== TestLib::Test ==================================================
    virtual unsigned int Init ();
    virtual unsigned int Start();
    virtual unsigned int Stop ();

};
