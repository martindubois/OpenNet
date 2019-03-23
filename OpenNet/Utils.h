
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Utils.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNet/Status.h>

// Functions
/////////////////////////////////////////////////////////////////////////////

OpenNet::Status Utl_ExceptionToStatus(KmsLib::Exception * aE);
