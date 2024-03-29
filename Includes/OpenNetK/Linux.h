
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Linux.h
/// \brief      (Linux)

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#ifdef __cplusplus
    #include <OpenNetK/LinuxCpp.h>
#endif

// Macros
/////////////////////////////////////////////////////////////////////////////

#ifdef _KMS_DEBUG_
    #define ASSERT(C) if ( ! ( C ) ) { printk( "Assert failed at line %d of %s - %s\n", __LINE__, __FILE__, #C ); }
#else
    #define ASSERT(C)
#endif

// TODO  OpenNetK.Linux
//       Normal (Feature) - Add TRACE_INFO and TRACE_WARNING

#define TRACE_DEBUG printk( KERN_DEBUG __NAMESPACE__ __CLASS__
#define TRACE_ERROR printk( KERN_ERR   __NAMESPACE__ __CLASS__

#define TRACE_END )

// Global variable
/////////////////////////////////////////////////////////////////////////////

#ifndef __cplusplus
    extern struct class * gOpenNet_Class      ;
    extern unsigned int   gOpenNet_DeviceCount;
#endif
