function FormSwitch({
  label,
  checked,
  onClick,
  children
}: {
  label: string;
  checked: boolean;
  onClick: (checked: boolean) => void;
  children?: React.ReactNode;
}) {
  return (
    <>
    <label className="inline-flex justify-between gap-24 cursor-pointer w-full">
        <span className="text-2xl">{label}</span>
        <input type="checkbox" value="" className="sr-only peer" checked={checked} onChange={(e) => onClick(e.target.checked)} />
        <div className="relative w-17 h-9 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 
        peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 
        peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full 
        peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] 
        after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full 
        after:w-8 after:h-8 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600 
        dark:peer-checked:bg-blue-600"></div>
    </label>
    {children && <p className="text-xs text-gray-500 text-wrap pl-4 w-5/6">{children}</p>}
    </>
  );
}

export default FormSwitch;