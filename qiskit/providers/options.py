# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Container class for backend options."""

import io
from collections.abc import Mapping


class Options(Mapping):
    """Base options object

    This class is what all backend options are based
    on. The properties of the class are intended to be all dynamically
    adjustable so that a user can reconfigure the backend on demand. If a
    property is immutable to the user (eg something like number of qubits)
    that should be a configuration of the backend class itself instead of the
    options.

    Instances of this class behave like dictionaries. Accessing an
    option with a default value can be done with the `get()` method:

    >>> options = Options(opt1=1, opt2=2)
    >>> options.get("opt1")
    1
    >>> options.get("opt3", default="hello")
    'hello'

    Key-value pairs for all options can be retrieved using the `items()` method:

    >>> list(options.items())
    [('opt1', 1), ('opt2', 2)]

    Options can be updated by name:

    >>> options["opt1"] = 3
    >>> options.get("opt1")
    3

    Runtime validators can be registered. See `set_validator`.
    Updates through `update_options` and indexing (`__setitem__`) validate
    the new value before performing the update and raise `ValueError` if
    the new value is invalid.

    >>> options.set_validator("opt1", (1, 5))
    >>> options["opt1"] = 4
    >>> options["opt1"]
    4
    >>> options["opt1"] = 10  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: ...
    """

    # Here there are dragons.

    # This class preamble is an abhorrent hack to make `Options` work similarly to a
    # SimpleNamespace, but with its instance methods and attributes in a separate namespace.  This
    # is required to make the initial release of Qiskit Terra 0.19 compatible with already released
    # versions of Qiskit Experiments, which rely on both of
    #       options.my_key = my_value
    #       transpile(qc, **options.__dict__)
    # working.
    #
    # Making `__dict__` a property which gets a slotted attribute solves the second line.  The
    # slotted attributes are not stored in a `__dict__` anyway, and `__slots__` classes suppress the
    # creation of `__dict__`.  That leaves it free for us to override it with a property, which
    # returns the options namespace `_fields`.
    #
    # We need to make attribute setting simply set options as well, to support statements of the
    # form `options.key = value`.  We also need to ensure that existing uses do not override any new
    # methods.  We do this by overriding `__setattr__` to purely write into our `_fields` dict
    # instead.  This has the highly unusual behavior that
    #       >>> options = Options()
    #       >>> options.validator = "my validator option setting"
    #       >>> options.validator
    #       {}
    #       >>> options.get("validator")
    #       "my validator option setting"
    # This is the most we can do to support the old interface; _getting_ attributes must return the
    # new forms where appropriate, but setting will work with anything.  All options can always be
    # returned by `Options.get`.  To initialise the attributes in `__init__`, we need to dodge the
    # overriding of `__setattr__`, and upcall to `object.__setattr__`.
    #
    # To support copying and pickling, we also have to define how to set our state, because Python's
    # normal way of trying to get attributes in the unpickle will fail.
    #
    # This is a terrible hack, and is purely to ensure that Terra 0.19 does not break versions of
    # other Qiskit-family packages that are already deployed.  It should be removed as soon as
    # possible.

    __slots__ = ("_fields", "validator")

    # implementation of the Mapping ABC:

    def __getitem__(self, key):
        return self._fields[key]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    # Allow modifying the options (validated)

    def __setitem__(self, key, value):
        self.update_options(**{key: value})

    # backwards-compatibility with Qiskit Experiments:

    @property
    def __dict__(self):
        return self._fields

    # SimpleNamespace-like access to options:

    def __getattr__(self, name):
        # This does not interrupt the normal lookup of things like methods or `_fields`, because
        # those are successfully resolved by the normal Python lookup apparatus.  If we are here,
        # then lookup has failed, so we must be looking for an option.  If the user has manually
        # called `self.__getattr__("_fields")` then they'll get the option not the full dict, but
        # that's not really our fault.  `getattr(self, "_fields")` will still find the dict.
        try:
            return self._fields[name]
        except KeyError as ex:
            raise AttributeError(f"Option {name} is not defined") from ex

    # setting options with the namespace interface is not validated
    def __setattr__(self, key, value):
        self._fields[key] = value

    # custom pickling:

    def __getstate__(self):
        return (self._fields, self.validator)

    def __setstate__(self, state):
        _fields, validator = state
        super().__setattr__("_fields", _fields)
        super().__setattr__("validator", validator)

    def __copy__(self):
        """Return a copy of the Options.

        The returned option and validator values are shallow copies of the originals.
        """
        out = self.__new__(type(self))  # pylint:disable=no-value-for-parameter
        out.__setstate__((self._fields.copy(), self.validator.copy()))
        return out

    def __init__(self, **kwargs):
        super().__setattr__("_fields", kwargs)
        super().__setattr__("validator", {})

    # The eldritch horrors are over, and normal service resumes below.  Beware that while
    # `__setattr__` is overridden, you cannot do `self.x = y` (but `self.x[key] = y` is fine).  This
    # should not be necessary, but if _absolutely_ required, you must do
    #       super().__setattr__("x", y)
    # to avoid just setting a value in `_fields`.

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self._fields.items())
        return f"{type(self).__name__}({', '.join(items)})"

    def __eq__(self, other):
        if isinstance(self, Options) and isinstance(other, Options):
            return self._fields == other._fields
        return NotImplemented

    def set_validator(self, field, validator_value):
        """Set an optional validator for a field in the options

        Setting a validator enables changes to an options values to be
        validated for correctness when :meth:`~qiskit.providers.Options.update_options`
        is called. For example if you have a numeric field like
        ``shots`` you can specify a bounds tuple that set an upper and lower
        bound on the value such as::

            options.set_validator("shots", (1, 4096))

        In this case whenever the ``"shots"`` option is updated by the user
        it will enforce that the value is >=1 and <=4096. A ``ValueError`` will
        be raised if it's outside those bounds. If a validator is already present
        for the specified field it will be silently overridden.

        Args:
            field (str): The field name to set the validator on
            validator_value (list or tuple or type): The value to use for the
                validator depending on the type indicates on how the value for
                a field is enforced. If a tuple is passed in it must have a
                length of two and will enforce the min and max value
                (inclusive) for an integer or float value option. If it's a
                list it will list the valid values for a field. If it's a
                ``type`` the validator will just enforce the value is of a
                certain type.
        Raises:
            KeyError: If field is not present in the options object
            ValueError: If the ``validator_value`` has an invalid value for a
                given type
            TypeError: If ``validator_value`` is not a valid type
        """

        if field not in self._fields:
            raise KeyError(f"Field '{field}' is not present in this options object")
        if isinstance(validator_value, tuple):
            if len(validator_value) != 2:
                raise ValueError(
                    "A tuple validator must be of the form '(lower, upper)' "
                    "where lower and upper are the lower and upper bounds "
                    "inclusive of the numeric value"
                )
        elif isinstance(validator_value, list):
            if len(validator_value) == 0:
                raise ValueError("A list validator must have at least one entry")
        elif isinstance(validator_value, type):
            pass
        else:
            raise TypeError(
                f"{type(validator_value)} is not a valid validator type, it "
                "must be a tuple, list, or class/type"
            )
        self.validator[field] = validator_value

    def update_options(self, **fields):
        """Update options with kwargs"""
        for field_name, field in fields.items():
            field_validator = self.validator.get(field_name, None)
            if isinstance(field_validator, tuple):
                if field > field_validator[1] or field < field_validator[0]:
                    raise ValueError(
                        f"Specified value for '{field_name}' is not a valid value, "
                        f"must be >={field_validator[0]} and <={field_validator[1]}"
                    )
            elif isinstance(field_validator, list):
                if field not in field_validator:
                    raise ValueError(
                        f"Specified value for {field_name} is not a valid choice, "
                        f"must be one of {field_validator}"
                    )
            elif isinstance(field_validator, type):
                if not isinstance(field, field_validator):
                    raise TypeError(
                        f"Specified value for {field_name} is not of required type {field_validator}"
                    )

        self._fields.update(fields)

    def __str__(self):
        no_validator = super().__str__()
        if not self.validator:
            return no_validator
        else:
            out_str = io.StringIO()
            out_str.write(no_validator)
            out_str.write("\nWhere:\n")
            for field, value in self.validator.items():
                if isinstance(value, tuple):
                    out_str.write(f"\t{field} is >= {value[0]} and <= {value[1]}\n")
                elif isinstance(value, list):
                    out_str.write(f"\t{field} is one of {value}\n")
                elif isinstance(value, type):
                    out_str.write(f"\t{field} is of type {value}\n")
            return out_str.getvalue()
