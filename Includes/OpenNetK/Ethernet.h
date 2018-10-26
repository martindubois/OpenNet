
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Ethernet.h

#pragma once

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This function returns a pointer to the Ethernet payload.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \return  This function returns a pointer to the Etehrnet payload.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne a pointeur vers la charge utile du
///          paquet Ethernet.
/// \param   aBase  Un pointeur vers le buffer
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \return  Cette fonction retourne a pointeur vers la charge utile du
///          paquet Ethernet.
/// \endcond
__global unsigned char * Ethernet_Data(__global unsigned char * aBase, __global const OpenNet_PacketInfo * aPacketInfo)
{
    return (aBase + aPacketInfo->mOffset_byte + 14);
}

/// \cond    en
/// \brief   This function returns Ethernet packet type.
/// \param   aBase        A pointer to the buffer
/// \param   aPacketInfo  A pointer to the information about the packet to
///                       process
/// \return  This function returns the Etehrnet packet type.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne le type du paquet Ethernet.
/// \param   aBase  Un pointeur vers le buffer
/// \param   aPacketInfo  Un pointeur vers l'information au sujet du paquet a
///          traiter
/// \return  Cette fonction retourne le type du paquet Ethernet.
/// \endcond
unsigned short Ethernet_Type(__global const unsigned char * aBase, __global const OpenNet_PacketInfo * aPacketInfo)
{
    return (*((__global const unsigned short *)(aBase + aPacketInfo->mOffset_byte + 12)));
}
