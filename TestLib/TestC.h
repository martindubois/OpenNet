
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       Common/TestLib/TestC.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Common =============================================================
#include "../Common/TestLib/Test.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class TestC : public TestLib::Test
{

public:

    TestC();

    // ===== TestLib::Test ==================================================
    virtual ~TestC();

    virtual void Info_Display() const;

protected:

    // ===== TestLib::Test ==================================================
    virtual void         AdjustGeneratorConfig(OpenNet::PacketGenerator::Config * aConfig);
    virtual unsigned int Init ();
    virtual unsigned int Start( unsigned int aFlags );
    virtual unsigned int Stop ();

private:

    OpenNet::UserBuffer * mUserBuffer;

};
