
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Function.h

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
        /// \return Cette methode retourne l'adresse d'une variable interne.
        /// \endcond
        OPEN_NET_PUBLIC const char * GetFunctionName() const;

        /// \cond en
        /// \brief  Set the function name
        /// \param  aFunctionName [---;R--] The function name
        /// \endcond
        /// \cond fr
        /// \brief  Modifier le nom de la fonction
        /// \param  aFunctionName [---;R--] Le nom de la fonction
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status SetFunctionName(const char * aFunctionName);

        // ===== SourceCode =================================================
        virtual        ~Function();
        virtual Status Display  (FILE * aOut) const;

    private:

        Function(const Function &);

        const Function & operator = (const Function &);

        char mFunctionName[64];

    };

}
