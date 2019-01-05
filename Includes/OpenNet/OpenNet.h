
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNet/OpenNet.h

#pragma once

// Macros
/////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32
    #ifdef OPENNET_EXPORTS
        #define OPEN_NET_PUBLIC __declspec( dllexport )
    #else
        #define OPEN_NET_PUBLIC __declspec( dllimport )
    #endif
#else
    #define OPEN_NET_PUBLIC
#endif
