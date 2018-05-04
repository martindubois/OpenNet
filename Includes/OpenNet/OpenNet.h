
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/OpenNet.h

#pragma once

// Macros
/////////////////////////////////////////////////////////////////////////////

#ifdef OPENNET_EXPORTS
    #define OPEN_NET_PUBLIC __declspec( dllexport )
#else
    #define OPEN_NET_PUBLIC __declspec( dllimport )
#endif
