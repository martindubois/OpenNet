
// Product / Produit  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/StdInt.h

#pragma once

// Data type
/////////////////////////////////////////////////////////////////////////////

#ifdef _KMS_LINUX_
    typedef unsigned int       uint32_t;
    typedef unsigned long long uint64_t;
    typedef unsigned char      uint8_t ;
#endif

#ifdef _KMS_WINDOWS_
    typedef unsigned __int32 uint32_t;
    typedef unsigned __int64 uint64_t;
    typedef unsigned __int8  uint8_t ;
#endif
