
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/OSDep.h
/// \brief      Define the OS dependent function the ONK_Lib calls.
/// \todo       Document functions

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

/// \cond en
/// \brief  Allocate non paged memory
/// \param  aSize_byte  The size of memory to allocate
/// \return The allocated memory's address
/// \endcond
/// \cond fr
/// \brief  Allouer de la memoire non pagine
/// \param  aSize_byte  La taille de la memoire a allouer
/// \return L'adresse de la memoire allouee
/// \endcond
/// \sa     OpenNetK_OSDep_FreeMemory
typedef void * ( * OpenNetK_OSDep_AllocateMemory )( unsigned int aSize_byte );

/// \cond en
/// \brief  Release non paged memory
/// \param  aMemory  The address of the memory to release
/// \endcond
/// \cond fr
/// \brief  Relacher de la memoire non pagine
/// \param  aMemory  L'adresse de la memoire a relacher
/// \endcond
/// \sa     OpenNetK_OSDep_AllocateMemory
typedef void ( * OpenNetK_OSDep_FreeMemory )( void * aMemory );

/// \cond en
/// \brief  Retrive a timestamp
/// \return A timestamp in us
/// \endcond
/// \cond fr
/// \brief  Obtenir un timestamp
/// \return Un timestamp en us
/// \endcond
typedef uint64_t ( * OpenNetK_OSDep_GetTimeStamp )( void );

/// \cond en
/// \brief  Lock a spinlock
/// \param  aLock  The spinlock
/// \note   This function is part of the critical path.
/// \endcond
/// \cond fr
/// \brief  Verouiller un spinlock
/// \param  aLock  Le spinlock
/// \note   Cette fonction fait partie du chemin critique
/// \endcond
/// \sa     OpenNetK_OSDep_UnlockSpinlock
typedef void  ( * OpenNetK_OSDep_LockSpinlock )( void * aLock );

/// \cond en
/// \brief  Lock a spinlock and disable interrupt
/// \param  aLock  The spinlock
/// \return A value to pass to OpenNetK_OSDep_UnlockSpinlockFromThread
/// \endcond
/// \cond fr
/// \brief  Verouiller un spinlock et desactiver les interruptions
/// \param  aLock  Le spinlock
/// \return Une valeur a passer a OpenNetK_OSDep_UnlockSpinlockFromThread
/// \endcond
/// \sa     OpenNetK_OSDep_UnlockSpinlockFromThread
typedef uint32_t ( * OpenNetK_OSDep_LockSpinlockFromThread  )( void * aLock );

/// \cond en
/// \brief  Unlock a spinlock
/// \param  aLock  The spinlock
/// \note   This function is part of the critical path.
/// \endcond
/// \cond fr
/// \brief  Deverouiller un spinlock
/// \param  aLock  Le spinlock
/// \note   Cette fonction fait partie du chemin critique
/// \endcond
/// \sa     OpenNetK_OSDep_LockSpinlock
typedef void ( * OpenNetK_OSDep_UnlockSpinlock )( void * aLock );

/// \cond en
/// \brief  Unlock a spinlock and restore interrupt
/// \param  aLock  The spinlock
/// \param  aFlags  The value OpenNetK_OSDep_LockSpinlockFromThread returned
/// \endcond
/// \cond fr
/// \brief  Deverouiller un spinlock et reactiver les interruptions
/// \param  aLock  Le spinlock
/// \param  aFlags  La valeur retournee par OpenNetK_OSDep_LockSpinlockFromThread
/// \endcond
/// \sa     OpenNetK_OSDep_LockSpinlockFromThread
typedef void ( * OpenNetK_OSDep_UnlockSpinlockFromThread)( void * aLock, uint32_t aFlags );

/// \cond en
/// \brief  Map a buffer
/// \param  aContext    The context
/// \param  aBuffer_PA  The input or output for the buffer physical address
/// \param  aBuffer_DA  The buffer device address
/// \param  aSize_byte  The size of the buffer
/// \param  aMarker_PA  The physical address of the marker
/// \param  aMarker_MA  The mapped address of the marker is returner here
/// \return This function return the mapped address of the buffer
/// \endcond
/// \cond fr
/// \brief  Mapper un buffer
/// \param  aContext    Le contexte
/// \param  aBuffer_PA  L'adresse physique du buffer en entree ou en sortie
/// \param  aBuffer_DA  L'adresse du buffer pour la carte graphique
/// \param  aSize_byte  La taille du buffer
/// \param  aMarker_PA  L'adresse physique du marqueur
/// \param  aMarker_MA  L'adresse a utiliser pour le marqueur est retournee
///         ici
/// \return Cette fonction retourne l'adresse a utiliser pour le buffer
/// \endcond
/// \sa     OpenNetK_OSDep_UnmapBuffer
typedef void * ( * OpenNetK_OSDep_MapBuffer )( void * aContext, uint64_t * aBuffer_PA, uint64_t aBuffer_DA, unsigned int aSize_byte, uint64_t aMarker_PA, volatile void * * aMarker_MA );

/// \cond en
/// \brief  Map a buffer
/// \param  aContext    The context
/// \param  aBuffer_MA  The mapped address of the buffer
/// \param  aSize_byte  The size of the buffer
/// \param  aMarker_MA  The mapped address of the marker is returner here
/// \endcond
/// \cond fr
/// \brief  Mapper un buffer
/// \param  aContext    Le contexte
/// \param  aBuffer_MA  L'adresse retournee par OpenNetK_OSDep_MapBuffer
/// \param  aSize_byte  La taille du buffer
/// \param  aMarker_MA  L'adresse a utiliser pour le marqueur
/// \endcond
/// \sa     OpenNetK_OSDep_MapBuffer
typedef void ( * OpenNetK_OSDep_UnmapBuffer )( void * aContext, void * aBuffer_MA, unsigned int aSize_byte, volatile void * aMarker_MA );

/// \cond en
/// \brief  Map the shared memory
/// \param  aContext    The context
/// \param  aShared_US  The user address of the shared memory
/// \param  aSize_byte  The size of the shared memory
/// \return This function return the mapped address of the shared memory
/// \endcond
/// \cond fr
/// \brief  Mapper la memoire partage
/// \param  aContext    Le contexte
/// \param  aShared_UA  L'adresse de la memoire partage en mode utilisateur
/// \param  aSize_byte  La taille de la memoire partage
/// \return Cette fonction retourne l'adresse a utiliser pour la memoire
///         partagee
/// \endcond
/// \sa     OpenNetK_OSDep_UnmapSharedMemory
typedef void * ( * OpenNetK_OSDep_MapSharedMemory )( void * aContext, void * aShared_UA, unsigned int aSize_byte );

/// \cond en
/// \brief  Unmap the shared memory
/// \param  aContext    The context
/// \endcond
/// \cond fr
/// \brief  Relacher la memoire partage
/// \param  aContext    Le contexte
/// \endcond
/// \sa     OpenNetK_OSDep_MapSharedMemory
typedef void ( * OpenNetK_OSDep_UnmapSharedMemory )( void * aContext );

/// \cond en
/// \brief  This structure contains pointer to OS dependant functions
/// \endcond
/// \cond fr
/// \brief  Cette structure contient des pointeurs vers les fonctions qui
///         dependes du systeme d'exploitation.
/// \endcond
/// \todo   Document members
typedef struct
{
    void * mContext;

    OpenNetK_OSDep_AllocateMemory AllocateMemory;
    OpenNetK_OSDep_FreeMemory     FreeMemory    ;

    OpenNetK_OSDep_GetTimeStamp GetTimeStamp;

    OpenNetK_OSDep_LockSpinlock             LockSpinlock            ;
    OpenNetK_OSDep_LockSpinlockFromThread   LockSpinlockFromThread  ;
    OpenNetK_OSDep_UnlockSpinlock           UnlockSpinlock          ;
    OpenNetK_OSDep_UnlockSpinlockFromThread UnlockSpinlockFromThread;

    OpenNetK_OSDep_MapBuffer   MapBuffer  ;
    OpenNetK_OSDep_UnmapBuffer UnmapBuffer;

    OpenNetK_OSDep_MapSharedMemory   MapSharedMemory  ;
    OpenNetK_OSDep_UnmapSharedMemory UnmapSharedMemory;

}
OpenNetK_OSDep;
