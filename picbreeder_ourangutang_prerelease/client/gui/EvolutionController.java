/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.gui;

import client.*;
import client.evolution.*;
import client.renderers.*;

import java.awt.event.*;

import javax.swing.*;

import com.sun.jmx.snmp.Timestamp;

import java.util.*;

import client.server.ServerConnectionInstance;
import client.server.ServerException;
import client.renderers.Renderer;

import java.io.ByteArrayOutputStream;

public class EvolutionController extends JPanel implements ActionListener, TaskListener, ItemListener {
	private final PopulationPanel population;
	private final Collection <JComponent> controls;
	private final JComponent back, forward, redo, save;
	private boolean redoEnabled, saveEnabled;
	private Renderer renderer;
	private final JPanel evoPanel, renderPanel;
	
	private static final Class [] NullClassArray = {};
	private static final Object [] NullObjectArray = {};
	
	private boolean registerRequired = false;
	public static Timestamp starting_timestamp;
	
	// for my own testing... ignore em
	long renderStartTime, renderEndTime;
	
	// quick hack for check box. it works
	private static final class RenderCheckBox extends JCheckBox implements ItemListener {
		private final client.renderers.RenderingAlgorithm selectedAlgorithm, deselectedAlgorithm;
		
		public RenderCheckBox() {
			super("High Quality Rendering", true);
			
			this.addItemListener(this);
			this.setToolTipText("Forces an image to completely render.");
			
			//selectedAlgorithm = new client.renderers.algorithms.LOD();
			selectedAlgorithm = new client.renderers.algorithms.RowBased();
			//deselectedAlgorithm = new client.renderers.algorithms.Background();
			deselectedAlgorithm = new client.renderers.algorithms.LowQualityTest();
			
			client.renderers.RenderingAlgorithmInstance.set(selectedAlgorithm);
		}
		
		public void itemStateChanged(ItemEvent e) {
			if(e.getStateChange() == ItemEvent.SELECTED)
				client.renderers.RenderingAlgorithmInstance.set(selectedAlgorithm);
			else if(e.getStateChange() == ItemEvent.DESELECTED)
				client.renderers.RenderingAlgorithmInstance.set(deselectedAlgorithm);
		}
	}
	
	public EvolutionController(PopulationPanel p) {
		super();
		
		 java.util.Date date= new java.util.Date();
		 starting_timestamp = new Timestamp(date.getTime());

		population = p;
		population.setEvolutionController(this);
		
		controls = new LinkedList <JComponent> ();
		
		evoPanel = new JPanel();
		renderPanel = new JPanel();
		
		evoPanel.setBorder(BorderFactory.createTitledBorder("Controls"));
		evoPanel.setLayout(new java.awt.FlowLayout());
		
		renderPanel.setBorder(BorderFactory.createTitledBorder("Rendering"));
		renderPanel.setLayout(new java.awt.FlowLayout());
		
		createControl("Quit",'\0',"toolbarButtonGraphics/general/Stop16.gif", false);
		//createControl("Restart", '\0', "toolbarButtonGraphics/navigation/Home16.gif").setToolTipText("Start over from the picture you branched off of.");
		back = createControl("Back", 'B', "toolbarButtonGraphics/navigation/Back16.gif");
		forward = createControl("Forward", 'F', "toolbarButtonGraphics/navigation/Forward16.gif");
		redo = createControl("Redo", 'R', "toolbarButtonGraphics/general/Redo16.gif");
		createControl("Spawn", '\0', "toolbarButtonGraphics/media/Play16.gif").setToolTipText("Select some pictures and click this to make more like them.");
		save = createControl("Save", 'S', "toolbarButtonGraphics/general/Save16.gif");
		createControl("Publish", 'P', "toolbarButtonGraphics/general/SaveAll16.gif").setToolTipText("Stop evolving and publish the selected image for the world to see.");
		
		back.setToolTipText("Go back one generation (if possible)");
		forward.setToolTipText("Go forward one generaion (if possible)");
		redo.setToolTipText("Respawn this generation with the last pictures picked");
		save.setToolTipText("Save your progress up to the current generation.");
		
		JComboBox renderMode = new JComboBox();
		
		if(CoreSpecificRenderer.isSupported())
			renderMode.addItem(new CoreSpecificRenderer(this));
		renderMode.addItem(new SingleRenderer(this));
		renderMode.addItem(new ParallelRenderer(this));
		
		renderMode.addItemListener(this);
		controls.add(renderMode);
		
		renderer = (Renderer) renderMode.getSelectedItem();
		
		JLabel renderLabel = new JLabel("Render Mode: ");
		renderLabel.setLabelFor(renderMode);
		renderPanel.add(renderLabel);
		renderPanel.add(renderMode);
		
		renderLabel.setToolTipText("Choose the mode that best suits your computer.");
		renderMode.setToolTipText("Choose the mode that best suits your computer.");
		
		JComponent renderBox = new RenderCheckBox();
		renderBox.setToolTipText("Higher quality rendering takes more time.");
		//controls.add(renderBox);
		renderPanel.add(renderBox);
		
		JButton refresh = new JButton("Refresh");
		refresh.setToolTipText("Refresh the current generation's images.");
		refresh.addActionListener(this);
		controls.add(refresh);
		renderPanel.add(refresh);
		
		back.setEnabled(false);
		forward.setEnabled(false);
		redo.setEnabled(true);
		save.setEnabled(true);
		redoEnabled = true;
		saveEnabled = true;

		
		this.setLayout(new java.awt.BorderLayout());
		//!this.add(evoPanel, java.awt.BorderLayout.CENTER);
		//this.add(renderPanel, java.awt.BorderLayout.SOUTH);
		
		try {
			updatePopulation();
		}
		catch(ServerException e) {
			// can't load!
			JOptionPane.showMessageDialog(MainComponentInstance.get(), e.getMessage(), e.getClass().getSimpleName(), JOptionPane.ERROR_MESSAGE);
			quit();
		}
	}
	

	private JComponent createControl(String name, char hotkey, String icon) {
		return createControl(name, hotkey, icon, true);
	}
	
	private JComponent createControl(String name, char hotkey, String icon, boolean disable) {
		JButton b = new JButton(name);
		b.addActionListener(this);
		b.setVerticalTextPosition(AbstractButton.BOTTOM);
		b.setHorizontalTextPosition(AbstractButton.CENTER);
		
		if(icon != null)
			b.setIcon(new ImageIcon(ImageDatabaseInstance.get().getImage(icon)));
		
		if(hotkey > '\0')
			b.setMnemonic(hotkey);
		
		if(disable)
			controls.add(b);
		evoPanel.add(b);
		return b;
	}
	
	public void updatePopulation() throws client.server.ServerException {
		Generation g = SeriesInstance.get().getCurrentGeneration();
		population.setGeneration(g);
		
		renderer.render(g.getIndividuals());
	}
	
	public void notifyTaskStarting() {
		renderStartTime = System.currentTimeMillis();
		
		for(JComponent comp : controls)
			comp.setEnabled(false);
	}
	
	public void notifyTaskFinished() {
		renderEndTime = System.currentTimeMillis();
		
		InformationPanel.setLastRenderTime(renderEndTime - renderStartTime);
		//System.out.println("Last Render Time: " + (renderEndTime - renderStartTime));
		
		for(JComponent comp : controls)
			comp.setEnabled(true);
		
		forward.setEnabled(SeriesInstance.get().canGoForward());
		back.setEnabled(SeriesInstance.get().canGoBack());
		redo.setEnabled(redoEnabled);
		save.setEnabled(saveEnabled);

		// try to gc in case we have been eating space.
		// this is a good downtime
		System.runFinalization();
		System.gc();
	}
	
	public void actionPerformed(ActionEvent event) {
		try {
			String command = event.getActionCommand().toLowerCase();
			getClass().getMethod(command, NullClassArray).invoke(this, NullObjectArray);
		}
		catch(java.lang.reflect.InvocationTargetException e2) {
			e2.printStackTrace();
			Exception e = (Exception) e2.getTargetException();
			JOptionPane.showMessageDialog(MainComponentInstance.get(), e.getMessage(), e.getClass().getSimpleName(), JOptionPane.ERROR_MESSAGE);
		}
		catch(Exception e) {
			// shouldn't happen!
			e.printStackTrace();
			JOptionPane.showMessageDialog(MainComponentInstance.get(), e.getMessage(), e.getClass().getSimpleName(), JOptionPane.ERROR_MESSAGE);
		}
	}
	
	public void itemStateChanged(ItemEvent event) {
		//multiThreadedRendering = event.getStateChange() == ItemEvent.SELECTED;
		if(event.getStateChange() == ItemEvent.SELECTED) {
			renderer = (Renderer) event.getItem();
		}
	}
	
	public void spawn() {
		try {

			
			redoEnabled = true;
			saveEnabled = true;
			SeriesInstance.get().spawn();
			updatePopulation();
		}
		catch(Exception e) {
			JOptionPane.showMessageDialog(MainComponentInstance.get(), e.getMessage(), e.getClass().getSimpleName(), JOptionPane.ERROR_MESSAGE);
		}
	}
	
	public void back() throws ServerException {
		if(SeriesInstance.get().canGoBack()) {
			try {
				SeriesInstance.get().goBack();
				updatePopulation();
				redoEnabled = false;
				saveEnabled = true;
			}
			catch(ServerException e) {
				SeriesInstance.get().goForward();
				throw e;
			}
		}
	}
	
	public void forward(){
		if(SeriesInstance.get().canGoForward()) {
			try {
				SeriesInstance.get().goForward();
				updatePopulation();
				redoEnabled = false;
				saveEnabled = true;
			}
			catch(ServerException e) {
				e.printStackTrace();
				// SHOULD NOT HAPPEN
			}
		}
	}
	
	// TODO cleaner code
	public void redo() throws ServerException {
		if(redoEnabled) {
			if(SeriesInstance.get().canGoBack()) {
				try {
					saveEnabled = true;
					SeriesInstance.get().goBack();
					SeriesInstance.get().spawn();
					updatePopulation();
				}
				catch(EvolutionException e) {
					// happens when no parents were selected in the previous generation
					// for now report it
					JOptionPane.showMessageDialog(MainComponentInstance.get(), "Fixed Bug: Redo not possible. Please report!", "Bug", JOptionPane.INFORMATION_MESSAGE);
					SeriesInstance.get().goForward();
				}
				catch(ServerException e) {
					// happens when spawn tries to spawn from a
					// previous generation which was not able to load.
					// the back method succeeds, the spawn doesn't.
					// to restore, we must go forward
					SeriesInstance.get().goForward();
				}
			}
			else {
				SeriesInstance.get().initializeFirstGeneration();
				updatePopulation();
			}
		}
	}
	
	public void save()
			throws client.server.ServerException, EvolutionException {
		//if(JOptionPane.showConfirmDialog(MainComponentInstance.get(), "Known Bug Workaround: Saving will close the applet. Are you sure you want to save now?", "Known Bug: Save Series", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE) == JOptionPane.NO_OPTION)
		//	return;
		
		saveToServer();
		if(registerRequired)
			((SessionEnd) MainComponentInstance.get()).register();
		//quit();
	}
	
	/**
	 * Attempts to log the user into the system. If the user decide to cancel the operation,
	 * the method returns <code>false</code> so that an error box will not pop up.
	 * 
	 * @return <code>true</code> if the user wants to continue, <code>false</code> otherwise.
	 * @throws client.server.LoginException The user information is bad
	 * @throws client.server.TimeoutException The server cannot be reached
	 */
	private boolean loginUser() throws client.server.LoginException, client.server.TimeoutException {
		LoginBox box = new LoginBox();
		boolean r = box.hasLoginInformation();
		
		try {
			if(r) {
				client.server.ServerConnectionInstance.get().logInExistingUser(box.getUserName(), box.getEncodedPassword());
				((SessionEnd)MainComponentInstance.get()).authenticate(box.getUserName(), box.getRawPassword());
			}
			else
				r = registerRequired = box.requiresRegistration();
		}
		catch(client.server.LoginException e) {
			throw e;
		}
		catch(client.server.TimeoutException e) {
			throw e;
		}
		finally {
			box.dispose();
		}
		
		return r;
	}
	
	private void saveToServer()
			throws client.server.FatalException, client.server.TimeoutException, client.server.LoginException, EvolutionException {
		Individual rep = null;
		
		try {
			rep = SeriesInstance.get().getCurrentGeneration().getRepresentative();
		}
		catch(ServerException e) {
			// won't happen
			e.printStackTrace();
		}
		
		if(rep == null)
			throw new EvolutionException("You must select a representative.");
		
		try {
			if(!client.server.ServerConnectionInstance.get().isUserLoggedIn())
				if(!loginUser())
					return;
			
			String name = client.server.ServerConnectionInstance.get().getSeriesName();
			SeriesInstance.get().setCurrentBranch(name);
		}
		catch(client.server.FatalException e) {
			// TODO handle and quit
			throw e;
		}
		
		// Timeout will be handled by the actionPerformed method
		// no it won't, i'm retarded... ignore me
		// yes it will, i'm twice as retarded!
		try {
			client.utilities.XML.store(SeriesInstance.get(), ServerConnectionInstance.get().getSaveStreamForSeries());
			client.utilities.XML.store(rep.getGenome(), ServerConnectionInstance.get().getSaveStreamForGenome());
			
			java.util.Map <String, Transferable> dataMap = DatabaseInstance.get().getAdditionList();
			
			String [] remove = {}, add = {};
			
			remove = DatabaseInstance.get().getRemovalList().toArray(remove);
			add = dataMap.keySet().toArray(add);
			
			for(String storage : add)
				client.utilities.XML.store(dataMap.get(storage), ServerConnectionInstance.get().getSaveStreamForStorage(storage));
	
			if(registerRequired) {
				if(remove.length > 0)
					throw new client.evolution.EvolutionException("Internal Registration Error. PLEASE REPORT THIS!");
				ServerConnectionInstance.get().saveAnonymously(add);
			}
			else
				ServerConnectionInstance.get().save(remove, add);
			
			SeriesInstance.get().notifySaveSuccessful();
		}
		// TODO better exception handling
		catch(java.io.IOException e) {
			throw new client.server.FatalException(e.getMessage());
		}
		
		saveEnabled = false;
		save.setEnabled(false);
		
	}
	
	public void publish() throws client.server.ServerException, EvolutionException {
		if(saveEnabled)
			saveToServer();
		
		if(registerRequired) {
			JOptionPane.showMessageDialog(MainComponentInstance.get(), "Please publish this image from your user panel after you confirm your account.", "Anonymous Publish", JOptionPane.INFORMATION_MESSAGE);
			((SessionEnd) MainComponentInstance.get()).register();
		}
		else	
			((SessionEnd) MainComponentInstance.get()).publish();
	}
	
	public void quit() {
		if(saveEnabled) {
			int res = JOptionPane.showConfirmDialog(MainComponentInstance.get(), "Quitting will destroy unsaved data. Are you sure you want to quit?", "Unsaved Data", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);
			
			if(res != JOptionPane.YES_OPTION)
				return;
		}
		
		((SessionEnd) MainComponentInstance.get()).quit();	
	}
	
	public void refresh() {
		try {
			updatePopulation();
		}
		catch(ServerException e) {
			// won't happen
		}
	}
}
