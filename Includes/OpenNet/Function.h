
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNet/Function.h
/// \brief      OpenNet::Function

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/SourceCode.h>

namespace OpenNet
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  The Function class
    /// \endcond
    /// \cond fr
    /// \brief  La classe Function
    /// \endcond
    class Function : public SourceCode
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \endcond
        OPEN_NET_PUBLIC Function();

        /// \cond en
        /// \brief  Retrieve the function name
        /// \return This method returns the address of an internal variable.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le nom de la fonction
        /// \return Cette m&eacute;thode retourne l'adresse d'une variable interne.
        /// \endcond
        /// \sa     SetFunctionName
        OPEN_NET_PUBLIC const char * GetFunctionName() const;

        /// \cond en
        /// \brief  Set the function name
        /// \param  aFunctionName  The function name
        /// \endcond
        /// \cond fr
        /// \brief  Modifier le nom de la fonction
        /// \param  aFunctionName  Le nom de la fonction
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        /// \sa     GetFunctionName
        virtual Status SetFunctionName(const char * aFunctionName);

        // ===== SourceCode =================================================
        OPEN_NET_PUBLIC virtual        ~Function();
        OPEN_NET_PUBLIC virtual Status Display  (FILE * aOut) const;

    private:

        Function(const Function &);

        const Function & operator = (const Function &);

        char mFunctionName[64];

    };

}
