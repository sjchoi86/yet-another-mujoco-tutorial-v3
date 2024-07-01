import tkinter as tk
import numpy as np

class MultiSliderClass(object):
    """
        GUI with multiple sliders
    """
    def __init__(self,
                 n_slider      = 10,
                 title         = 'Multiple Sliders',
                 window_width  = 500,
                 window_height = None,
                 x_offset      = 500,
                 y_offset      = 100,
                 slider_width  = 400,
                 label_texts   = None,
                 slider_mins   = None,
                 slider_maxs   = None,
                 slider_vals   = None,
                 resolution    = None,
                 resolutions   = None,
                 verbose       = True
        ):
        """
            Initialze multiple sliders
        """
        self.n_slider      = n_slider
        self.title         = title
        
        self.window_width  = window_width
        if window_height is None:
            self.window_height = self.n_slider*40
        else:
            self.window_height = window_height
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.slider_width  = slider_width
        
        self.resolution    = resolution
        self.resolutions   = resolutions
        self.verbose       = verbose
        
        # Slider values
        self.slider_values = np.zeros(self.n_slider)
        
        # Initial/default slider settings
        self.label_texts   = label_texts
        self.slider_mins   = slider_mins
        self.slider_maxs   = slider_maxs
        self.slider_vals   = slider_vals
        
        # Create main window
        self.gui = tk.Tk()
        
        self.gui.title("%s"%(self.title))
        self.gui.geometry(
            "%dx%d+%d+%d"%
            (self.window_width,self.window_height,self.x_offset,self.y_offset))
        
        # Create vertical scrollbar
        self.scrollbar = tk.Scrollbar(self.gui,orient=tk.VERTICAL)
        
        # Create a Canvas widget with the scrollbar attached
        self.canvas = tk.Canvas(self.gui,yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure the scrollbar to control the canvas
        self.scrollbar.config(command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a frame inside the canvas to hold the sliders
        self.sliders_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0),window=self.sliders_frame,anchor=tk.NW)
        
        # Create sliders
        self.sliders = self.create_sliders()
        
        # Update the canvas scroll region when the sliders_frame changes size
        self.sliders_frame.bind("<Configure>",self.cb_scroll)

        # You may want to do this in the main script
        for _ in range(100): self.update() # to avoid GIL-related error 
        
    def cb_scroll(self,event):    
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def cb_slider(self,slider_idx,slider_value):
        """
            Slider callback function
        """
        self.slider_values[slider_idx] = slider_value # append
        if self.verbose:
            print ("slider_idx:[%d] slider_value:[%.1f]"%(slider_idx,slider_value))
        
    def create_sliders(self):
        """
            Create sliders
        """
        sliders = []
        for s_idx in range(self.n_slider):
            # Create label
            if self.label_texts is None:
                label_text = "Slider %02d "%(s_idx)
            else:
                label_text = "[%d/%d]%s"%(s_idx,self.n_slider,self.label_texts[s_idx])
            slider_label = tk.Label(self.sliders_frame, text=label_text)
            slider_label.grid(row=s_idx,column=0,padx=0,pady=0)
            
            # Create slider
            if self.slider_mins is None: slider_min = 0
            else: slider_min = self.slider_mins[s_idx]
            if self.slider_maxs is None: slider_max = 100
            else: slider_max = self.slider_maxs[s_idx]
            if self.slider_vals is None: slider_val = 50
            else: slider_val = self.slider_vals[s_idx]

            # Resolution
            if self.resolution is None: # if none, divide the range with 100
                resolution = (slider_max-slider_min)/100
            else:
                resolution = self.resolution 
            if self.resolutions is not None:
                resolution = self.resolutions[s_idx]

            slider = tk.Scale(
                self.sliders_frame,
                from_      = slider_min,
                to         = slider_max,
                orient     = tk.HORIZONTAL,
                command    = lambda value,idx=s_idx:self.cb_slider(idx,float(value)),
                resolution = resolution,
                length     = self.slider_width
            )
            slider.grid(row=s_idx,column=1,padx=0,pady=0,sticky=tk.W)
            slider.set(slider_val)
            sliders.append(slider)
            
        return sliders
    
    def update(self):
        if self.is_window_exists():
            self.gui.update()
        
    def run(self):
        self.gui.mainloop()
        
    def is_window_exists(self):
        try:
            return self.gui.winfo_exists()
        except tk.TclError:
            return False
        
    def get_slider_values(self):
        return self.slider_values
    
    def set_slider_values(self,slider_values):
        self.slider_values = slider_values
        for slider,slider_value in zip(self.sliders,self.slider_values):
            slider.set(slider_value)

    def set_slider_value(self,slider_idx,slider_value):
        self.slider_values[slider_idx] = slider_value
        slider = self.sliders[slider_idx]
        slider.set(slider_value)
    
    def close(self):
        if self.is_window_exists():
            # some loop
            for _ in range(100): self.update() # to avoid GIL-related error 
            # Close 
            self.gui.destroy()
            self.gui.quit()
            self.gui.update()