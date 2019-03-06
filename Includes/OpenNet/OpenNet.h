
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
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
    // TODO  OpenNet
    //       Normal - On Linux, Do not export all symbols
    #define OPEN_NET_PUBLIC
#endif
