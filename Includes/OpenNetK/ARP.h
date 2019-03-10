
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/ARP.h

#pragma once

// Constants
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   The Ethernet type of ARP packets
/// \endcond
/// \cond    fr
/// \brief   Le type Ethernet des paquets ARP
/// \endcond
#define ARP_ETHERNET_TYPE (0x0608)

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   This function returns a pointer to the requested address.
/// \param   aData  A pointer to the ARP header
/// \return  This function returns a pointer to the requested address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse
///          questionn&eacute;e.
/// \param   aData  Un pointeur vers l'ent&ecirc;te ARP
/// \return  Cette fonction retourne un pointeur vers l'adresse
///          questionn&eacute;e.
/// \endcond
__global unsigned char * ARP_Destination(__global unsigned char * aData)
{
    return (aData + 24);
}

/// \cond    en
/// \brief   This function returns a pointer to the source address.
/// \param   aData  A pointer to the ARP header
/// \return  This function returns a pointer to the source address.
/// \endcond
/// \cond    fr
/// \brief   Cette fonction retourne un pointeur vers l'adresse de
///          provenance.
/// \param   aData  Un pointeur vers l'ent&ecirc;te ARP
/// \return  Cette fonction retourne un pointeur vers l'adresse de
///          provenance.
/// \endcond
__global unsigned char * ARP_Source(__global unsigned char * aData)
{
    return (aData + 14);
}
