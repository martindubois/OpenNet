
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/IPv4.h
///
/// \cond    en
/// This file defines constant and functions used to manipulate IPv4 packets.
/// \endcond
/// \cond    fr
/// Ce fichier defini une constante et des fonction utilise pour manipuler des
/// paquets IPv4.
/// \endcond

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   The Ethernet type of IPv4 packets
/// \endcond
/// \cond    fr
/// \brief   Le type Ethernet des paquets IPv4
/// \endcond
#define IPv4_ETHERNET_TYPE (0x0008)

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This function returns a pointer to the destination address.
/// \param   aData  A pointer to the IPv4 header
/// \return  This function returns a pointer to the destination address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne a pointeur vers l'adresse de
///          destination.
/// \param   aData  Un pointeur vers l'entete IPv4
/// \return  Cette fonction retourne a pointeur vers l'adresse de
///          destination.
/// \endcond
__global unsigned char * IPv4_Destination(__global unsigned char * aData)
{
    return (aData + 16);
}

/// \cond    en
/// \brief   This function returns a pointer to the source address.
/// \param   aData  A pointer to the IPv4 header
/// \return  This function returns a pointer to the source address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne a pointeur vers l'adresse de provenance.
/// \param   aData  Un pointeur vers l'entete IPv4
/// \return  Cette fonction retourne a pointeur vers l'adresse de provenance.
/// \endcond
__global unsigned char * IPv4_Source(__global unsigned char * aData)
{
    return (aData + 12);
}
