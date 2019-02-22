
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/System_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "System_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class System_CUDA : public System_Internal
{

public:

    System_CUDA();

    // ====== OpenNet::System ===============================================

    virtual ~System_CUDA();

private:

    void FindAdapters  ();
    void FindProcessors();

};
