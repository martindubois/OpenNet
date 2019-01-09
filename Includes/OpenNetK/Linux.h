
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Linux.h
/// \brief      Define what is needed on Linux

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef long unsigned int size_t;

// Constantes
/////////////////////////////////////////////////////////////////////////////

#define _IOC_NONE  ( 0 )
#define _IOC_READ  ( 2 )
#define _IOC_WRITE ( 1 )

#define _IOC_NRBITS   (  8 )
#define _IOC_TYPEBITS (  8 )
#define _IOC_SIZEBITS ( 14 )
#define _IOC_DIRBITS  (  2 )

#define _IOC_NRSHIFT   ( 0 )
#define _IOC_TYPESHIFT ( _IOC_NRSHIFT   + _IOC_NRBITS  )
#define _IOC_SIZESHIFT ( _IOC_TYPESHIFT + _IOC_TYPEBITS )
#define _IOC_DIRSHIFT  ( _IOC_SIZESHIFT + _IOC_SIZEBITS )

#define KERN_SOH "\001"

#define KERN_EMERG   KERN_SOH "0"
#define KERN_ALERT   KERN_SOH "1"
#define KERN_CRIT    KERN_SOH "2"
#define KERN_ERR     KERN_SOH "3"
#define KERN_WARNING KERN_SOH "4"
#define KERN_NOTICE  KERN_SOH "5"
#define KERN_INFO    KERN_SOH "6"
#define KERN_DEBUG   KERN_SOH "7"

#define KERN_DEFAULT KERN_SOH "d"

#define NULL ( 0 )

// Macros
/////////////////////////////////////////////////////////////////////////////

#define ASSERT(C) if ( ! ( C ) ) { printk( "Assert failed at line %d of $s - %s\n", __LINE__, __FILE__, #C ); }

#define _IOC(D,T,N,S) static_cast< unsigned int >( ( (D) << _IOC_DIRSHIFT ) | ( (T) << _IOC_TYPESHIFT ) | ( (N) << _IOC_NRSHIFT ) | ( (S) << _IOC_SIZESHIFT ) ) 

#define _IO(T,N)        _IOC( _IOC_NONE             , (T), (N),           0 )
#define _IOR(T,N,S)     _IOC( _IOC_READ             , (T), (N), sizeof( S ) )
#define _IOW(T,N,S)     _IOC( _IOC_WRITE            , (T), (N), sizeof( S ) )
#define _IOW_BAD(T,N,S) _IOC( _IOC_READ             , (T), (N), sizeof( S ) )
#define _IOWR(T,N,S)    _IOC( _IOC_READ | _IOC_WRITE, (T), (N), sizeof( S ) )

#define SIZE_OF(S) static_cast< uint32_t >( sizeof( S ) )

// Function
/////////////////////////////////////////////////////////////////////////////

extern "C"
{
    
    void * memcpy( void * aOut, const void * aIn, size_t aSize_byte );
    void * memset( void * aPtr, int aValue, size_t aSize_byte );
    
    int printk( const char * aFormat, ... );
    
    char * strcpy( char * aOut, const char * aIn );
    
}
