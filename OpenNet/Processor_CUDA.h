
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor_CUDA.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet ============================================================
#include "Processor_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_CUDA : public Processor_Internal
{

public:

    Processor_CUDA( int aDevice, KmsLib::DebugLog * aDebugLog );

    // ====== Processor_Internal ============================================

    virtual ~Processor_CUDA();

private:

    void InitInfo();

};
