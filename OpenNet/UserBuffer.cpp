
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/UserBuffer.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNet/UserBuffer.h>

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void UserBuffer::Delete()
    {
        delete this;
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    UserBuffer::UserBuffer()
    {
    }

    UserBuffer::~UserBuffer()
    {
    }

}
