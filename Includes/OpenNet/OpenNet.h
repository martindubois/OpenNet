
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
    #define OPEN_NET_PUBLIC_CLASS
#else
    #ifdef OPENNET_EXPORTS
        #define OPEN_NET_PUBLIC         __attribute__ ( ( visibility ( "default" ) ) )
        #define OPEN_NET_PUBLIC_CLASS   __attribute__ ( ( visibility ( "default" ) ) )
    #else
        #define OPEN_NET_PUBLIC
        #define OPEN_NET_PUBLIC_CLASS
    #endif
#endif
