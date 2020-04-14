
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2020 KMS. All rights reserved.
// Product    OpenNet
// File       TestLib/TestG.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "../Common/TestLib/Test.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class TestG : public TestLib::Test
{

public:

    TestG();

    // ===== TestLib::Test ==================================================
    virtual ~TestG();

    virtual void Info_Display() const;

protected:

    // ===== TestLib::Test ==================================================
    virtual unsigned int Init();
    virtual unsigned int Start(unsigned int aFlags);
    virtual unsigned int Stop();

};
