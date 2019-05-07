
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/ByteOrder.h
/// \brief      ByteOrder_Swap16, ByteOrder_Swap32 (RT)

// TEST COVERAGE  2019-05-03  KMS - Martin Dubois, ing.

#pragma once

// Functions
/////////////////////////////////////////////////////////////////////////////

/// \cond    en
/// \brief   Reverse byte order
/// \param   aIn  The input value
/// \return  This method returns the input value with reversed byte order
/// \endcond
/// \cond    fr
/// \brief   Renverse l'ordre des octets
/// \param   aIn  La valeur d'entr&eacute;e
/// \return  Cette m&eacute;thode retourne la valeur d'entr&eacute;e avec
///          l'ordre des octet invers&eacute;
/// \endcond
unsigned short ByteOrder_Swap16(unsigned short aIn)
{
    return ((aIn >> 8) | (aIn << 8));
}

/// \cond    en
/// \brief   Reverse byte order
/// \param   aIn  The input value
/// \return  This method returns the input value with reversed byte order
/// \endcond
/// \cond    fr
/// \brief   Renverse l'ordre des octets
/// \param   aIn  La valeur d'entr&eacute;e
/// \return  Cette m&eacute;thode retourne la valeur d'entr&eacute;e avec
///          l'ordre des octet invers&eacute;
/// \endcond
unsigned int ByteOrder_Swap32(unsigned int aIn)
{
    return ((aIn >> 24) | ((aIn >> 8) & 0x0000ff00) | ((aIn << 8) & 0x00ff0000) | (aIn << 24));
}
