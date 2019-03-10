
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/OS.h

#pragma once

// Includes
//////////////////////////////////////////////////////////////////////////////

// ====== Includes ===========================================================

#ifdef _KMS_LINUX_
    #include <OpenNetK/Linux.h>
#endif

#ifdef _KMS_WINDOWS_
    #include <OpenNetK/Windows.h>
#endif
